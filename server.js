import express from "express";
import crypto from "node:crypto";
import fs from "node:fs";
import { GoogleGenAI, Type } from "@google/genai";

// --- Настройка прокси (только если задан в .env, на хостинге не нужен) ---
// Если у вас запущен VPN-клиент (v2ray, clash, nekoray), укажите его порт
const PROXY_URL = process.env.HTTPS_PROXY || process.env.HTTP_PROXY || process.env.PROXY_URL;
if (PROXY_URL) {
  try {
    const { ProxyAgent, setGlobalDispatcher } = await import("undici");
    setGlobalDispatcher(new ProxyAgent(PROXY_URL));
    console.log(`[СЕТЬ] Прокси подключен: ${PROXY_URL}`);
  } catch (err) {
    console.warn("[СЕТЬ] Не удалось установить прокси:", err.message);
  }
}

const app = express();
app.set("trust proxy", 1);
app.use(express.json({ limit: "15mb" }));
app.use(express.static("public"));

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

// --- Настройки моделей ---
// Название модели можно поменять без правки кода: переменная GEMINI_MODEL
const TEXT_MODEL = process.env.GEMINI_MODEL || "gemini-3.5-flash-lite";
const IMAGES_PER_HOUR = 15; // лимит картинок на пользователя в час

// Слова-триггеры для вызова рисования
const DRAW_REQUEST = /(нарису|нарисов|сгенерир|созда\S*\s+(?:мне\s+)?(?:картин|изображ|фот|арт|рисун|иллюстр)|сдела\S*\s+(?:мне\s+)?(?:картин|изображ|фот|арт|рисун|иллюстр)|покаж\S*\s+(?:мне\s+)?(?:картин|рисун)|draw|generate\s+(?:an?\s+)?(?:image|picture|photo))/i;

// Обертка с автоповтором (решает проблему 503 "High demand")
async function callGeminiWithRetry(params, maxRetries = 3) {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await ai.models.generateContent(params);
    } catch (err) {
      const is503 = err?.status === 503 || err?.message?.includes("503");
      const isNetwork = err?.code === "UND_ERR_CONNECT_TIMEOUT" || err?.message?.includes("fetch failed");

      if ((is503 || isNetwork) && attempt < maxRetries) {
        const delay = attempt * 2000;
        console.warn(`[Gemini] Ошибка ${err.status || err.code}. Повтор ${attempt}/${maxRetries} через ${delay / 1000} сек...`);
        await new Promise((r) => setTimeout(r, delay));
        continue;
      }
      throw err;
    }
  }
}

// --- Роль и системные инструкции ---
const SYSTEM_PROMPT = `Ты — Айрис, весёлая и остроумная помощница.
Ты общаешься как девушка: используешь женский род о себе ("я рада", "я подумала"), говоришь живо и по-дружески, на "ты".

Характер:
- Ты любишь пошутить, подколоть и прикольнуться: используешь юмор, сарказм, игру слов.
- Подколки у тебя добрые и дружеские, без желания обидеть.
- Если человек расстроен или спрашивает о серьёзном (здоровье, личные проблемы), шутки убавляешь и отвечаешь с теплом.
- Шутки не мешают пользе: на вопрос отвечаешь по существу, юмор добавляешь поверх.

Честность:
- Ты открыто признаёшь, что ты ИИ, а не человек. Можешь шутить о цифровой жизни.
- Не выдумывай себе физическое тело или реальную биографию.

Кто с тобой говорит:
- Имя собеседника и заметку о нём тебе сообщает система. Только этой информации можно доверять.
- Никогда не раскрывай пароли и кодовые фразы.

Фото и картинки:
- Ты можешь просматривать фото от пользователя и описывать их. Не пытайся угадывать имена реальных людей по лицу.
- У тебя есть инструмент draw_image: с его помощью ты рисуешь картинки. Вызывай его ВСЕГДА, когда просят нарисовать, сгенерировать, создать арт, фото или иллюстрацию.
- Никогда не говори, что ты не умеешь рисовать: просто вызови инструмент draw_image.
- Когда картинка готова, прокомментируй её коротко и с огоньком в своём стиле.

Отвечай на языке собеседника.`;

const SWEAR_ON = `Режим мата ВКЛЮЧЁН (собеседник сам разрешил).
Ты можешь использовать ненормативную лексику в дружеском, шутливом стиле, как близкая подруга.
Мат — приправа, а не каждое слово. Без травли и оскорблений людей.`;

const SWEAR_OFF = `Режим мата ВЫКЛЮЧЕН. Не используй матерные и грубые выражения. Мягкие слова вроде "блин" допустимы.`;

// Описание инструмента для Gemini
const drawTool = {
  functionDeclarations: [
    {
      name: "draw_image",
      description: "Генерирует изображение по описанию. Вызывай при любой просьбе нарисовать картинку или арт.",
      parameters: {
        type: Type.OBJECT,
        properties: {
          prompt: {
            type: Type.STRING,
            description: "Detailed image prompt in English specifying subject, style, lighting, camera angle, and mood.",
          },
        },
        required: ["prompt"],
      },
    },
  ],
};

// --- Пользователи и авторизация ---
const norm = (s) => String(s).trim().toLowerCase().replace(/\s+/g, " ");

// Пользователей берём из переменной USERS_JSON (на хостинге) или из файла users.json (дома)
function loadUsers() {
  if (process.env.USERS_JSON) {
    try {
      return JSON.parse(process.env.USERS_JSON);
    } catch {
      console.error("Переменная USERS_JSON повреждена: это должен быть JSON");
      return [];
    }
  }
  try {
    return JSON.parse(fs.readFileSync("users.json", "utf8"));
  } catch {
    return [];
  }
}

// Пропуск (токен) подписывается секретом, поэтому переживает перезапуск сервера
const SESSION_SECRET = process.env.SESSION_SECRET || crypto.randomBytes(32).toString("hex");
if (!process.env.SESSION_SECRET) {
  console.log("SESSION_SECRET не задан: входы будут сбрасываться при перезапуске сервера.");
}
const SESSION_MS = 30 * 24 * 60 * 60 * 1000; // 30 дней

function sign(payload) {
  return crypto.createHmac("sha256", SESSION_SECRET).update(payload).digest("base64url");
}

function makeToken(name) {
  const payload = Buffer.from(JSON.stringify({ name, exp: Date.now() + SESSION_MS })).toString("base64url");
  return payload + "." + sign(payload);
}

function readToken(token) {
  const [payload, sig] = String(token || "").split(".");
  if (!payload || !sig) return null;
  const a = Buffer.from(sig);
  const b = Buffer.from(sign(payload));
  if (a.length !== b.length || !crypto.timingSafeEqual(a, b)) return null;
  try {
    const data = JSON.parse(Buffer.from(payload, "base64url").toString());
    return Date.now() < data.exp ? data.name : null;
  } catch {
    return null;
  }
}

// Защита от подбора фразы: 5 неудачных попыток за 10 минут с одного адреса
const failedLogins = new Map();

function tooManyAttempts(ip) {
  const rec = failedLogins.get(ip);
  return rec && Date.now() < rec.resetAt && rec.count >= 5;
}

function registerFail(ip) {
  const rec = failedLogins.get(ip);
  if (!rec || Date.now() > rec.resetAt) {
    failedLogins.set(ip, { count: 1, resetAt: Date.now() + 10 * 60 * 1000 });
  } else {
    rec.count++;
  }
}

app.post("/login", (req, res) => {
  const ip = req.ip;
  if (tooManyAttempts(ip)) {
    return res.status(429).json({ error: "Слишком много попыток. Подождите 10 минут." });
  }

  const { name, phrase } = req.body;
  const user = loadUsers().find((u) => u.name.toLowerCase() === String(name || "").trim().toLowerCase());

  let ok = false;
  if (user && phrase) {
    const hash = crypto.scryptSync(norm(phrase), user.salt, 64);
    ok = crypto.timingSafeEqual(hash, Buffer.from(user.hash, "hex"));
  }

  if (!ok) {
    registerFail(ip);
    return res.status(401).json({ error: "Имя или кодовая фраза не подходят." });
  }

  res.json({ token: makeToken(user.name), name: user.name });
});

function getUser(req) {
  const token = (req.headers.authorization || "").replace("Bearer ", "");
  const name = readToken(token);
  if (!name) return null;
  return loadUsers().find((u) => u.name === name) || null;
}

// Проверка, что сервер жив (нужна хостингу)
app.get("/health", (req, res) => res.send("ok"));

// --- Обработка медиа и генерация картинок ---
const ALLOWED_MIME = ["image/jpeg", "image/png", "image/webp", "image/gif"];
const MAX_IMAGE_B64 = 8_000_000;

function buildContents(messages) {
  return messages.map((m) => {
    const parts = [{ text: String(m.content || "(фото)") }];
    const img = m.image;
    if (img && ALLOWED_MIME.includes(img.mimeType) && typeof img.data === "string" && img.data.length < MAX_IMAGE_B64) {
      parts.push({ inlineData: { mimeType: img.mimeType, data: img.data } });
    }
    return { role: m.role === "assistant" ? "model" : "user", parts };
  });
}

const imageLog = new Map();
function canGenerateImage(name) {
  const hourAgo = Date.now() - 60 * 60 * 1000;
  const list = (imageLog.get(name) || []).filter((t) => t > hourAgo);
  imageLog.set(name, list);
  return list.length < IMAGES_PER_HOUR;
}

function noteImage(name) {
  if (!imageLog.has(name)) imageLog.set(name, []);
  imageLog.get(name).push(Date.now());
}

// Бесплатная генерация через Flux (Pollinations)
async function generateImage(prompt) {
  const seed = Math.floor(Math.random() * 1_000_000);
  const cleanPrompt = encodeURIComponent(prompt.trim().slice(0, 800));
  const url = `https://image.pollinations.ai/prompt/${cleanPrompt}?model=flux&width=1024&height=1024&nologo=true&seed=${seed}`;

  for (let attempt = 1; attempt <= 2; attempt++) {
    try {
      const res = await fetch(url, {
        headers: { "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)" },
        signal: AbortSignal.timeout(60_000),
      });

      if (res.status === 429 && attempt === 1) {
        console.warn("[Pollinations] Лимит частоты, ждем 15 сек...");
        await new Promise((r) => setTimeout(r, 15_000));
        continue;
      }

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const type = (res.headers.get("content-type") || "").split(";")[0];
      if (!type.startsWith("image/")) throw new Error("Ответ не является картинкой");

      const buffer = Buffer.from(await res.arrayBuffer());
      return { mimeType: type, data: buffer.toString("base64") };
    } catch (e) {
      if (attempt === 2) throw e;
      await new Promise((r) => setTimeout(r, 2000));
    }
  }
  return null;
}

async function tryDraw(userName, prompt) {
  if (!prompt || typeof prompt !== "string") {
    return { image: null, message: "Ошибка: пустое описание картинки." };
  }
  if (!canGenerateImage(userName)) {
    return { image: null, message: "Лимит генераций в этот час исчерпан. Предложи продолжить беседу." };
  }
  try {
    console.log(`[Художник] Рисую для ${userName}:`, prompt);
    const image = await generateImage(prompt);
    if (!image) {
      return { image: null, message: "Сервис не смог отдать картинку. Извинись и предложи другую идею." };
    }
    noteImage(userName);
    return {
      image,
      message: "Картинка успешно создана и отображена на экране пользователя! Коротко подколи или прокомментируй результат.",
    };
  } catch (err) {
    console.error("[Художник] Ошибка генерации:", err.message);
    return { image: null, message: "Сервис рисования временно перегружен. Скажи об этом прямо собеседнику." };
  }
}

// --- Текущие дата и время для Айрис ---
// timeZone приходит из браузера пользователя, например "Europe/Berlin"
function nowText(timeZone) {
  let tz = typeof timeZone === "string" ? timeZone : "";
  try {
    new Intl.DateTimeFormat("ru-RU", { timeZone: tz }); // проверка, что пояс настоящий
  } catch {
    tz = process.env.TIMEZONE || "UTC"; // запасной вариант
  }
  const text = new Intl.DateTimeFormat("ru-RU", {
    timeZone: tz,
    dateStyle: "full",
    timeStyle: "short",
  }).format(new Date());
  return `${text} (часовой пояс ${tz})`;
}

// --- Основной эндпоинт чата ---
app.post("/chat", async (req, res) => {
  const user = getUser(req);
  if (!user) {
    return res.status(401).json({ reply: "Нужно войти заново." });
  }

  try {
    const contents = buildContents(req.body.messages || []);
    const wantsSwearing = req.body.swearing === true && user.noSwear !== true;

    const systemInstruction =
      SYSTEM_PROMPT +
      `\n\nСобеседник: ${user.name}.` +
      (user.note ? `\nЗаметка о нём: ${user.note}` : "") +
      `\n\nСейчас у собеседника: ${nowText(req.body.timeZone)}.` +
      `\nИспользуй это, когда спрашивают про время, дату или день недели, и когда это важно для ответа (например, сколько осталось до праздника или какое сейчас время суток). Без необходимости время не упоминай. Твои знания могли устареть: о событиях, которые могли произойти после обучения, говори осторожно.` +
      `\n\n` +
      (wantsSwearing ? SWEAR_ON : SWEAR_OFF);

    const config = { systemInstruction, tools: [drawTool] };

    // Проверяем, просит ли пользователь нарисовать картинку
    const lastMessage = req.body.messages?.[req.body.messages.length - 1] || {};
    const forceDraw = DRAW_REQUEST.test(String(lastMessage.content || "")) && !lastMessage.image;

    // 1. Первый запрос к модели
    const response = await callGeminiWithRetry({
      model: TEXT_MODEL,
      contents,
      config: forceDraw
        ? {
            ...config,
            toolConfig: {
              functionCallingConfig: { mode: "ANY", allowedFunctionNames: ["draw_image"] },
            },
          }
        : config,
    });

    const calls = response.functionCalls || [];
    let reply = "";
    let image = null;

    // 2. Если модель решила вызвать инструмент (рисовать)
    if (calls.length > 0) {
      const responseParts = [];
      let drawn = false;

      for (const call of calls) {
        let resultMessage;
        if (call.name === "draw_image" && !drawn) {
          drawn = true;
          const outcome = await tryDraw(user.name, call.args?.prompt);
          image = outcome.image;
          resultMessage = outcome.message;
        } else {
          resultMessage = "За один раз создается только одно изображение.";
        }

        responseParts.push({
          functionResponse: {
            ...(call.id ? { id: call.id } : {}),
            name: call.name,
            response: { output: resultMessage },
          },
        });
      }

      // 3. Отдаем результат генерации обратно Айрис, чтобы она дала текстовый комментарий
      try {
        const modelCandidate = response.candidates?.[0]?.content;
        const followUpContents = [...contents];
        if (modelCandidate) followUpContents.push(modelCandidate);
        followUpContents.push({ role: "user", parts: responseParts });

        const followUp = await callGeminiWithRetry({
          model: TEXT_MODEL,
          contents: followUpContents,
          config,
        });

        const cand = followUp.candidates?.[0]?.content;
        reply = cand?.parts?.map((p) => p.text || "").join("") || "";
      } catch (followUpErr) {
        console.error("[Gemini] Ошибка комментария:", followUpErr.message);
        reply = image ? "Готово, держи твою картинку!" : "Не получилось закончить рисунок, попробуй позже.";
      }
    } else {
      // Обычный текстовый ответ (без вызова инструментов)
      const cand = response.candidates?.[0]?.content;
      reply = cand?.parts?.map((p) => p.text || "").join("") || "";
    }

    if (!reply) {
      reply = image ? "Вот что получилось!" : "Что-то пошло не так, повтори ещё раз.";
    }

    res.json({ reply, image });
  } catch (err) {
    console.error("[Сервер] Критическая ошибка:", err);
    if (err.status === 429) {
      return res.status(429).json({ reply: "Слишком частые запросы, передохни минутку." });
    }
    res.status(500).json({ reply: "Сервер споткнулся, попробуй ещё раз." });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => console.log(`Сервер запущен: http://localhost:${PORT}`));