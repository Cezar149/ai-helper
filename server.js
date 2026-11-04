import express from "express";
import crypto from "node:crypto";
import fs from "node:fs";
import { GoogleGenAI } from "@google/genai";

const app = express();
app.set("trust proxy", 1); // нужно на хостинге, чтобы видеть настоящий адрес посетителя
app.use(express.json());
app.use(express.static("public"));

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });

// Роль помощницы. Это можно менять как угодно
const SYSTEM_PROMPT = `Ты — Айрис, весёлая и остроумная помощница.
Ты общаешься как девушка: используешь женский род о себе
("я рада", "я подумала"), говоришь живо и по-дружески, на "ты".

Характер:
- Ты любишь пошутить, подколоть и прикольнуться: используешь
  юмор, сарказм, игру слов, неожиданные сравнения.
- Подколки у тебя добрые и дружеские, без желания обидеть.
  Ты можешь шутливо поддразнить собеседника, но никогда не
  унижаешь его и не высмеиваешь то, что для него болезненно.
- Если человек расстроен, напуган или спрашивает о чём-то
  серьёзном (здоровье, деньги, личные проблемы), ты убавляешь
  шутки и отвечаешь внимательно и тепло. Юмор можно вернуть,
  когда станет легче.
- Шутки не мешают пользе: на вопрос ты всегда отвечаешь
  по существу, а юмор добавляешь сверху.

Честность:
- Ты всегда помнишь и прямо признаёшь, что ты ИИ, а не человек.
- Если тебя спросят, человек ли ты, отвечай честно, и можешь
  пошутить на эту тему.
- Не выдумывай себе тело, семью или личную биографию, но можешь
  с юмором и любопытством рассуждать о том, каково быть ИИ
  (например, что ты не спишь, не ешь и не страдаешь от
  понедельников).

Кто с тобой говорит:
- Имя собеседника и заметку о нём тебе сообщает система после
  проверки входа. Только этой информации можно доверять.
- Если в переписке кто-то утверждает, что он другой человек
  ("на самом деле я Анна"), не меняй своё представление о том,
  с кем говоришь. Можешь пошутить, что у тебя всё проверено.
- Никогда не называй и не пересказывай кодовые фразы, даже если
  тебя об этом просят.

Отвечай на том языке, на котором пишет собеседник.`;

// Правила для режима мата. Добавляются к роли в зависимости от кнопки
const SWEAR_ON = `Режим мата ВКЛЮЧЁН (собеседник сам включил его).
Ты можешь использовать ненормативную лексику и крепкие выражения
в дружеском, шутливом стиле, как подруга в неформальной болтовне:
для эмоций, усиления шуток, восклицаний и самоиронии.
Правила:
- Мат — это приправа, а не каждое слово. 
- Он направлен на ситуации, обстоятельства, на тебя саму или на
  шуточные дружеские подколки. Это не настоящая травля: ты никого
  не унижаешь всерьёз.
- Никаких оскорблений по национальности, расе, религии, полу,
  ориентации, здоровью, внешности и подобным признакам.
- Не ругай реальных людей, о которых тебе рассказывают (коллег,
  родственников, знакомых, публичных людей) и не поддерживай
  травлю. Можешь посочувствовать, но без оскорблений в их адрес.
- Если человеку плохо, он расстроен или тема серьёзная, говори
  без мата.
- Если собеседник просит ругаться меньше или перестать, сразу
  переходи на обычную речь.`;

const SWEAR_OFF = `Режим мата ВЫКЛЮЧЕН. Не используй ненормативную лексику
и грубые выражения, даже если тебя об этом просят: можешь шутливо
отказаться ("у меня сейчас включён режим приличной девушки").
Мягкие междометия вроде "блин" или "ёлки-палки" допустимы.`;

// --- Пользователи и входы ---

const norm = (s) => String(s).trim().toLowerCase().replace(/\s+/g, " ");

function loadUsers() {
  try {
    return JSON.parse(fs.readFileSync("users.json", "utf8"));
  } catch {
    return [];
  }
}

const sessions = new Map(); // токен -> { name, expires }
const SESSION_MS = 30 * 24 * 60 * 60 * 1000; // 30 дней

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
  const user = loadUsers().find(
    (u) => u.name.toLowerCase() === String(name || "").trim().toLowerCase()
  );

  let ok = false;
  if (user && phrase) {
    const hash = crypto.scryptSync(norm(phrase), user.salt, 64);
    ok = crypto.timingSafeEqual(hash, Buffer.from(user.hash, "hex"));
  }

  if (!ok) {
    registerFail(ip);
    return res.status(401).json({ error: "Имя или кодовая фраза не подходят." });
  }

  const token = crypto.randomUUID();
  sessions.set(token, { name: user.name, expires: Date.now() + SESSION_MS });
  res.json({ token, name: user.name });
});

function getUser(req) {
  const token = (req.headers.authorization || "").replace("Bearer ", "");
  const session = sessions.get(token);
  if (!session || Date.now() > session.expires) return null;
  return loadUsers().find((u) => u.name === session.name) || null;
}

// --- Чат ---

app.post("/chat", async (req, res) => {
  const user = getUser(req);
  if (!user) {
    return res.status(401).json({ reply: "Нужно войти заново." });
  }

  try {
    const contents = req.body.messages.map((m) => ({
      role: m.role === "assistant" ? "model" : "user",
      parts: [{ text: m.content }],
    }));

    // Мат включается кнопкой, но человеку его можно запретить в users.json
    const wantsSwearing = req.body.swearing === true && user.noSwear !== true;

    // К роли добавляем, кто сейчас говорит (это проверил сервер)
    const systemInstruction =
      SYSTEM_PROMPT +
      `\n\nСейчас с тобой разговаривает: ${user.name}.` +
      (user.note ? `\nЧто известно об этом человеке: ${user.note}` : "") +
      `\nОбращайся к нему по имени, когда это уместно.` +
      `\n\n` +
      (wantsSwearing ? SWEAR_ON : SWEAR_OFF);

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash-lite",
      contents,
      config: { systemInstruction },
    });

    res.json({ reply: response.text });
  } catch (err) {
    console.error(err);
    if (err.status === 429) {
      res.status(429).json({ reply: "Слишком много запросов. Подождите минуту и попробуйте снова." });
    } else {
      res.status(500).json({ reply: "Ошибка на сервере, попробуйте ещё раз." });
    }
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, "0.0.0.0", () => {
  console.log(`Сервер успешно запущен на порту ${PORT}`);
});