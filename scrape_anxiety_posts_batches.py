from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time

# CONFIG: Batch range
start_index = 0
end_index = 50

# Load anxiety threads CSV
threads_df = pd.read_csv("beyondblue_anxiety_threads_all.csv")
batch_df = threads_df.iloc[start_index:end_index]

# Chrome options
options = Options()
# options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--window-size=1920,1080")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

posts_data = []

for idx, row in batch_df.iterrows():
    url = row["link"]
    print(f"\nProcessing [{idx}] {url}")

    driver.get(url)

    try:
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "div.lia-message-subject"))
        )
    except Exception as e:
        print(f"⚠️ Error loading {url}: {e}")
        continue

    soup = BeautifulSoup(driver.page_source, "html.parser")

    # Title
    title_elem = soup.select_one("div.lia-message-subject")
    title = title_elem.get_text(strip=True) if title_elem else ""

    # Content
    content_elem = soup.select_one("#bodyDisplay > div > p")
    content = content_elem.get_text(strip=True) if content_elem else ""

    # Author
    author_elem = soup.select_one("#link_8 > span")
    author = author_elem.get_text(strip=True) if author_elem else "Anonymous"

    # User ID
    user_id_elem = soup.select_one("a.lia-user-name-link")
    user_id = user_id_elem.get("href", "").split("/")[-1] if user_id_elem else ""

    # Date
    date_elem = soup.select_one(
        "div.lia-message-post-date > span > span"
    )
    date = date_elem.get("title", "").split(" ")[0] if date_elem else ""

    # Post ID (from URL)
    post_id = url.split("-")[-1]

    # Post Category (Anxiety)
    post_category = "Anxiety"

    # Number of Comments
    comments_count_elem = soup.select_one("span.lia-component-messages-count")
    num_comments = comments_count_elem.get_text(strip=True) if comments_count_elem else "0"

    posts_data.append({
        "Post ID": post_id,
        "Post Title": title,
        "Post Content": content,
        "Post Author": author,
        "User ID": user_id,
        "Post Date": date,
        "Post Category": post_category,
        "Number of Comments": num_comments
    })

    time.sleep(1)

driver.quit()

# Save batch
df = pd.DataFrame(posts_data)
output_file = f"beyondblue_anxiety_posts_{start_index}_{end_index}.csv"
df.to_csv(output_file, index=False)

print(f"\n✅ Scraped {len(df)} posts successfully.")
print(f"💾 Saved to: {output_file}")
