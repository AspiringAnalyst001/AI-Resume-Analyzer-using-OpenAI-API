🧠 AI Resume Analyzer using OpenAI API & LangChain


[https://github.com/AspiringAnalyst001/AI-Resume-Analyzer-using-OpenAI-API/blob/main/Project%20Overview.png](https://github.com/AspiringAnalyst001/AI-Resume-Analyzer-using-OpenAI-API/blob/main/Project%20Overview.png)






An AI-powered Resume Analyzer built with Streamlit, LangChain, and OpenAI API that intelligently evaluates resumes, matches them against job descriptions, and provides actionable insights with semantic and AI-based analysis.

🚀 Features

✅ Upload and extract text from PDF resumes
✅ Paste or upload Job Descriptions (JD) for analysis
✅ Compute TF-IDF & Semantic Similarity using embeddings
✅ Generate GPT-driven resume feedback and improvement tips
✅ Detect missing skills and keywords automatically
✅ Download Markdown reports with insights
✅ Built-in Chroma Vector Database for semantic search
✅ Secure API handling via Streamlit Secrets

🧰 Tech Stack
Component	Technology
Frontend	Streamlit
Backend	Python
AI/LLM Framework	LangChain
Embeddings	OpenAI text-embedding-3-small
Vector Store	ChromaDB
Model	GPT-4o-mini
Deployment	Streamlit Cloud
⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/yourusername/ai-resume-analyzer.git
cd ai-resume-analyzer

2️⃣ Create and Activate a Virtual Environment
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Add Your OpenAI API Key Securely

Create a .streamlit/secrets.toml file (never commit this file!):

OPENAI_API_KEY = "sk-your-api-key"


⚠️ On Streamlit Cloud, go to App → Settings → Secrets and paste the key there instead.

▶️ Run the Application
streamlit run app.py


Once running, open http://localhost:8501
 in your browser.

🧮 How It Works

1️⃣ Extract Text → Reads and cleans text from PDF resumes & job descriptions
2️⃣ Embed & Compare → Uses OpenAI embeddings to measure semantic similarity
3️⃣ TF-IDF Scoring → Calculates quantitative similarity between resume & JD
4️⃣ Chroma Search → Finds top semantic matches across text chunks
5️⃣ GPT Analysis → Generates tailored improvement feedback and keyword insights
6️⃣ Report Generation → Creates a downloadable Markdown analysis report

🧠 Example Output
Metric	Example Result
TF-IDF Similarity	83.45%
Semantic Match	89.22%
Missing Keywords	AWS, API Integration, Agile
GPT Feedback	“Strong data skills, but needs more emphasis on cloud deployment experience.”
📦 Project Structure
ai-resume-analyzer/
│
├── app.py                 # Main Streamlit app
├── utils.py               # Helper functions (TF-IDF, text extraction, etc.)
├── requirements.txt       # Dependencies
├── .streamlit/
│   └── secrets.toml       # OpenAI API key (not pushed to GitHub)
└── chroma_db/             # Auto-generated local vector store (in-memory in cloud)

🌐 Deployment on Streamlit Cloud

1️⃣ Push your project to GitHub
2️⃣ Go to Streamlit Cloud

3️⃣ Select your repo and branch
4️⃣ Add your OpenAI API key in App → Settings → Secrets
5️⃣ Deploy 🚀

The app automatically uses in-memory ChromaDB for cloud compatibility.

🧩 Future Enhancements

🧾 Add multi-resume comparison

🧠 Incorporate ATS score prediction

📈 Visualize resume strengths with charts

🔗 Integrate with LinkedIn or job boards

☁️ Store user history in a lightweight database

👨‍💻 Author

Daipayan Sengupta
💼 Data & Automation Enthusiast | MIS & Dashboard Developer
📍 Passionate about data intelligence, automation, and GenAI applications
🔗 Connect on LinkedIn

🧾 License

This project is licensed under the MIT License
.

💡 “AI won’t replace humans, but humans using AI will replace those who don’t.”



