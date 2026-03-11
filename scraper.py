from newspaper import Article
from PIL import Image
import requests
from io import BytesIO

def get_scraped_data(url):
    """Downloads article text and the lead image."""
    article = Article(url)
    article.download()
    article.parse()
    
    # Combine headline and body text
    full_text = f"{article.title}\n\n{article.text}"
    
    # Ensure image is RGB for the 16-expert ensemble
    img = Image.new('RGB', (224, 224), color=(255, 255, 255))
    if article.top_image:
        try:
            resp = requests.get(article.top_image, timeout=10)
            img = Image.open(BytesIO(resp.content)).convert("RGB")
        except:
            pass
            
    return {"text": full_text, "image": img}