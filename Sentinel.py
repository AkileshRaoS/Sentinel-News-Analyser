from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
genai.configure(api_key="")
import os
import matplotlib.pyplot as plt
import io
import base64
import contextlib
from markdown import markdown
import re
import io
import os
import requests
import base64
import contextlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from flask import Flask, render_template, request
from markdown import markdown
import matplotlib
import matplotlib.pyplot as plt





# Set the directory to save plots
PLOT_DIR = 'static/images'

# Ensure matplotlib does not try to open windows
matplotlib.use('Agg')

app = Flask(__name__)

# Function to fetch the full content from an article's URL
def fetch_full_content(url, timeout=20):
    try:
        response = requests.get(url, timeout=timeout)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        full_content = ' '.join([para.get_text() for para in paragraphs])
        return full_content
    except (requests.exceptions.RequestException, requests.exceptions.Timeout):
        return None

# Function to fetch news using the NewsAPI
def fetch_news(api_key, query, from_date=None, to_date=None, language='en', sort_by='relevancy', limit=10):
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': query,
        'from': from_date,
        'to': to_date,
        'language': language,
        'sortBy': sort_by,
        'apiKey': api_key,
        'pageSize': limit
    }
    response = requests.get(url, params=params)
    return response.json()


def analyze_keywords(article_content):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(f'''
    Extract the most relevant keyword(s) (maximum 2 words) that best represent the core theme of the following news article. 
    The keywords should be specific, context-aware, and suitable for a precise news search.
    
    Article:
    {article_content}

    Respond with only the keyword(s) without any explanation.
    ''')
    return response.text.strip()




def extract_articles(news_data):
    articles = news_data.get('articles', [])
    extracted_data = []
    for article in articles[:10]:
        content = article.get('content', '')
        if '[+' in content:
            full_content = fetch_full_content(article['url'])
        else:
            full_content = content

        publisher_name = article.get('source', {}).get('name', 'Unknown Publisher')
        extracted_data.append({
            'title': article['title'],
            'description': article['description'],
            'content': full_content if full_content else article['description'],
            'url': article['url'],
            'published_at': article['publishedAt'],
            'publisher': publisher_name
        })
    return extracted_data

# Function to fetch content concurrently from URLs
def fetch_full_content_concurrently(articles, timeout=20):
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_article = {executor.submit(fetch_full_content, article['url'], timeout): article for article in articles}
        for future in as_completed(future_to_article):
            article = future_to_article[future]
            try:
                full_content = future.result()
                if full_content:
                    article['content'] = full_content
                else:
                    article['content'] = article['description']
            except Exception:
                article['content'] = article['description']
    return articles

# Function to interact with the Gemini model to analyze sentiment and bias
def analyze_with_gemini(articles):
    combined_content = ""
    for article in articles:
        combined_content += f"Title: {article['title']}\n"
        combined_content += f"Publisher: {article['publisher']}\n"
        combined_content += f"Content: {article['content']}\n\n"
    

    prompt = (
    f"Here are multiple news articles:\n\n"
    f"{combined_content}\n\n"
    "Task: You are a helpful and accurate assistant. Help me with my bias detection for news articles by conducting both a linguistic bias analysis and a sentiment analysis. Use the formulas provided below to calculate standardized scores on a 0–10 scale for each article:\n\n"
    "Standardization Formulas:\n"
    "1. Linguistic Bias Score = (B + G + E) / T × 100\n"
    "   - Where:\n"
    "     - B = Number of biased terms identified\n"
    "     - G = Number of group-specific references\n"
    "     - E = Number of exclusionary phrases\n"
    "     - T = Total word count\n"
    "   - Convert to a 0–10 scale by dividing the result by 10.\n\n"
    "2. Sentiment Bias Score = (P + N + I) / T × 100\n"
    "   - Where:\n"
    "     - P = Number of personal pronouns (I, we, you, etc.)\n"
    "     - N = Number of charged/emotional words\n"
    "     - I = Number of intensifiers (very, extremely, etc.)\n"
    "     - T = Total word count\n"
    "   - Convert to a 0–10 scale by dividing the result by 10.\n\n"
    "For each article, perform the following analysis:\n\n"
    "1. **Sentiment Analysis**:\n"
    "   - *Sentiment Score:* [X]/10 (Score ranges from 0 (highly negative) to 10 (highly positive)).\n"
    "   - *Explanation:* Provide a detailed explanation by identifying specific words, phrases, or sentences that contribute to the article’s tone. Include key examples that illustrate the emotional slant of the content.\n\n"
    "2. **Linguistic Bias Analysis**:\n"
    "   - *Bias Score:* [X]/10 (Score ranges from 0 (least biased) to 10 (highly biased)).\n"
    "   - *Explanation:* Describe any biased language, framing, selective focus, or overgeneralizations present in the article. Provide detailed examples such as key phrases, adjectives, or framing techniques that indicate bias.\n\n"
    "For each article, structure your analysis as follows:\n\n"
    "### Title: **[Article Title]**\n"
    "### Publisher: **[Publisher Name]**\n\n"
    "## 1. Sentiment Analysis:\n"
    "*Sentiment Score:* [X]/10\n"
    "*Explanation:* [Explain the score by referencing specific quotes or examples that reveal the tone of the article.]\n\n"
    "## 2. Linguistic Bias Analysis:\n"
    "*Bias Score:* [X]/10\n"
    "*Explanation:* [Provide a detailed breakdown of any biases detected, with key examples.]\n\n"
    "## 3. Comparative Analysis:\n"
    "- Compare this article with the others in terms of neutrality, objectivity, and the range of perspectives covered.\n"
    "- Identify aspects or viewpoints that each article may be missing relative to the others.\n\n"
    "After analyzing each article individually, provide the following summary:\n\n"
    "### Summary of Scores:\n"
    "1. **Linguistic Bias Scores**:\n"
    "   - List each article’s bias score along with its publisher.\n"
    "2. **Sentiment Scores**:\n"
    "   - List each article’s sentiment score along with its publisher.\n\n"
    "### Average Scores:\n"
    "- Provide the average sentiment score across all articles.\n"
    "- Provide the average linguistic bias score across all articles.\n\n"
    "Finally, generate a Python code snippet to create vertically aligned histograms for the sentiment and linguistic bias scores with the following guidelines:\n\n"
    "1. The x-axis should display the publisher names.\n"
    "2. Plot Sentiment Scores in the top histogram and Linguistic Bias Scores in the bottom histogram.\n"
    "3. Set the y-axis maximum value to 10 for both histograms.\n"
    "4. Rotate the x-axis labels by 45 degrees and align them to the right using `set_xticklabels()` with `ha='right'`.\n"
    "5. Increase the figure size to improve readability (e.g., `figsize=(10, 12)` or larger).\n"
    "6. Ensure the histograms are vertically stacked, with one above the other.\n\n"
    "Important: Ensure that every part of your response is accurate and based solely on the provided content. Do not introduce external assumptions or inaccuracies."
    "If any article is empty neglect that article don't add it in analysis"
)



    


    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    render_template('index.html', analysis_text=None, combined_content=combined_content)

    return response.text

# Function to execute Python code and save plot
def execute_python_code(code):
    local_vars = {}
    with contextlib.redirect_stdout(io.StringIO()) as stdout:
        try:
            exec(code, {}, local_vars)
        except Exception as e:
            return str(e), None

    output = stdout.getvalue()
    plot_path = None
    if 'plt' in local_vars:  # Check if matplotlib was used
        # Create a unique plot file name
        plot_filename = 'plot.png'
        plot_path = os.path.join(PLOT_DIR, plot_filename)
        
        # Save the plot
        local_vars['plt'].savefig(plot_path)
        local_vars['plt'].close()  # Close the plot to prevent showing

    return output, plot_path


@app.route('/rewrite', methods=['POST'])
def rewrite():
    combined_content = request.form.get('combined_content', '')
    analysis_text = request.form['analysis_text']

    prompt = (
    f"Here are the original news articles:\n\n"
    f"{combined_content}\n\n"
    f"Here is the analysis of potential bias and missing perspectives:\n\n"
    f"{analysis_text}\n\n"
    "### Task:\n"
    "Write a single, comprehensive news article based on the provided articles and analysis. Ensure the article presents a balanced view by incorporating missing perspectives highlighted in the analysis. The article should read like a professional news report.\n\n"
    "### Structure:\n"
    "- **Headline:**\n"
    "  - The headline should be bold and in a larger font, appearing on a separate line and it should be catchy but without bias.\n"
    "- **Content:**\n"
    "  - Structure the article in clear paragraphs.\n"
    "  - Incorporate key points from the original articles and integrate the missing perspectives naturally.\n"
    "  - Ensure a neutral and factual tone, avoiding phrases like 'as noted in the analysis' or 'according to this source.'\n"
    "  - Present the facts as part of a cohesive story.\n\n"
    "### Guidelines:\n"
    "- **Headline Format:** The headline should appear on a new line and be bold with a larger font.\n"
    "- **Tone:** Maintain a neutral, factual tone throughout the article.\n"
    "- **Paragraph Structure:** Present information in clear paragraphs with smooth transitions.\n"
    "- **Quotes:** Use relevant quotes from the original articles, but integrate them seamlessly into the narrative.\n"
    )

    # Interact with the Gemini model
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    rewritten_text = markdown(response.text)
    
    return render_template('index.html', analysis_text=None, rewritten_text=rewritten_text)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        api_key = ''  # Replace with your NewsAPI key
        query = request.form.get("query")
        
        # Fetch news articles
        news_data = fetch_news(api_key, query)
        articles = extract_articles(news_data)
        
        # Fetch full content for each article
        articles = fetch_full_content_concurrently(articles)
        print(articles)
        # Analyze articles using the Gemini model
        gemini_analysis = analyze_with_gemini(articles)

        k = gemini_analysis.split("```")
        
        n=k[0]
        analysis_text = n
        python_code = k[1].replace("python", "") if len(k) > 1 else "No code found."
        output, plot_path = execute_python_code(python_code)
        
        # print(python_code)
        analysis_text_html = markdown(analysis_text)
        return render_template("index.html", analysis_text=analysis_text_html, plot_path=plot_path, code_output=output)
    
    elif request.args.get('link'):
        link = request.args.get('link')
        
        # Step 1: Fetch the content from the provided URL
        article_content = fetch_full_content(link)
        if not article_content:
            analysis_text = "Failed to fetch content from the URL."
        else:
            # Step 2: Extract keywords from the article content
            keywords = analyze_keywords(article_content)
            keywords=keywords.split('\n')[0]
            print(keywords)

            # Step 3: Use keywords to fetch related articles
            api_key = ''  # Replace with your NewsAPI key
            news_data = fetch_news(api_key, keywords)
            
            # Step 4: Extract articles and fetch content concurrently
            articles = extract_articles(news_data)
            articles = fetch_full_content_concurrently(articles)
            gemini_analysis = analyze_with_gemini(articles)
            k = gemini_analysis.split("```")
            n=k[0]
            analysis_text = n
            python_code = k[1].replace("python", "") if len(k) > 1 else "No code found."
            output, plot_path = execute_python_code(python_code)
        
            # print(python_code)
            analysis_text_html = markdown(analysis_text)
            return render_template("index.html", analysis_text=analysis_text_html, plot_path=plot_path, code_output=output)

    return render_template("index.html", analysis_text=None, plot_path=None, code_output=None)

if __name__ == "__main__":
    # Create PLOT_DIR if it doesn't exist
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)
    
    app.run(debug=True)

