# 🌾 Kisaan Dost — Hackathon Demo Script (FAST NUCES)

**Time Limit:** 6 Minutes
**Prize:** PKR 1,500,000
**Track:** 3/5

---

## 🎤 INTRODUCTION (30 Seconds)

**Speaker:**
"Assalam-o-Alaikum Judges! Humari team aapko present kar rahi hai **Kisaan Dost** — Pakistan ka pehla Multi-Agent Agricultural Assistant. 
Pakistan mein 45 million farming families hain, aur har saal lag bhag Rs. 2,000 crore ki fasal preventable diseases se kharab hoti hai. 
Humara system kisan ko unki apni zubaan (Roman Urdu) mein disease diagnosis, irrigation scheduling, aur live market prices provide karta hai, using multiple specialized AI agents."

---

## 🧪 SCENARIO 1: DISEASE DETECTION WITH IMAGE (2 Minutes)

**Setup:** 
- Open the Kisaan Dost Streamlit App.
- Upload a photo of wheat with yellow stripes (pre-downloaded "wheat yellow rust disease Pakistan").

**Action:**
Type in the chat: 
> *"Yeh mere gandum mein kya hua hai? Faisalabad mein hun"*

**Expected System Flow (Point this out as it loads):**
1. Orchestrator detects the disease query and image, routing to **Crop Doctor**.
2. Vision model (Gemini-2.0) analyzes the image and identifies "Yellow Rust (Zard Zang)".
3. RAG pulls exact treatment from the agricultural database.

**What to Say to Judges:**
*"Yahaan aap dekh sakte hain — kisan ne sirf photo upload ki aur system ne automatically disease detect ki. System ne sirf aam advice nahi di, balke treatment suggest ki, LOCAL BRAND names (jaise Tilt 250EC) diye, aur estimated cost (Rs. 800-1200 per acre) bhi batayi. Ye vision AI aur RAG ka powerful combination hai jo specifically Pakistan ke context ke liye fine-tune kiya gaya hai."*

---

## 💧 SCENARIO 2: IRRIGATION + WEATHER (90 Seconds)

**Action:**
Type in the chat:
> *"Mujhe batao ke aaj chawal ko paani dena chahiye? Lahore mein hun"*

**Expected System Flow:**
1. Orchestrator routes to **Irrigation Advisor**.
2. Live weather API (OpenWeatherMap) fetches Lahore's real-time temperature and rainfall forecast.
3. RAG fetches rice water requirements.

**What to Say to Judges:**
*"Ab hum real-time weather data ka demo dekhte hain. OpenWeatherMap API se live data aa raha hai. System automatically calculate karta hai evapotranspiration aur fasal ki water zaroorat. Is advice ki madad se kisan electricity aur diesel bacha sakta hai, jo ke practically Rs. 800-1200 bachata hai har irrigation cycle mein."*

---

## 💰 SCENARIO 3: MARKET PRICE (60 Seconds)

**Action:**
Type in the chat:
> *"Aaj Lahore mein pyaaz ka kya bhav hai? 50 maund bikna hai"*

**Expected System Flow:**
1. Orchestrator routes to **Market Price Agent**.
2. Tavily API performs a live web search for today's mandi prices.
3. System calculates transportation cost and net earnings, then gives selling advice.

**What to Say to Judges:**
*"Ye real-time market intelligence hai. Kisan abhi tak andazay ya middle-men par depend karta tha. Humarey paas live mandi data hai, system automatically 50 maund ke hisaab se transportation cost calculate karta hai, aur net earning bhi batata hai. Ye directly unki income badhata hai."*

---

## 🧠 TECHNICAL TALKING POINTS (60 Seconds)

*(Keep these points sharp and confident)*

1. **Multi-Agent Architecture:** "Orchestrator decides which specialist agents to call based on the query type. Sab kuch parallel mein process hota hai."
2. **RAG Database:** "We indexed agricultural documents covering 11 major Pakistani crops, specifically tailored to our local soil and diseases."
3. **API Integration:** "Three free APIs working together seamlessly: OpenRouter vision, Groq text (Llama-3.3-70b), OpenWeatherMap weather, and Tavily market search."
4. **Localization:** "Roman Urdu support — because that's how 90% of Pakistani farmers actually type on their phones."
5. **Scalability:** "Runs on zero-cost APIs initially — this can be deployed at scale for Pakistan's 45 million farming families with minimal overhead."

---

## 🚀 IMPACT STATEMENT / CLOSING (30 Seconds)

**Speaker:**
"Pakistan mein agriculture backbone hai, magar kisan akela hai. Humarey system se kisan apne phone se — Roman Urdu mein — crop disease diagnose kar sakta hai, irrigation schedule hasil kar sakta hai, aur live market prices check kar sakta hai. 

Ye sirf technology nahi hai — ye Pakistan ke kisan ki zaroorat hai, taake 'Kisaan Dost' waqai unka sab se behtar dost ban sake. Thank you!"
