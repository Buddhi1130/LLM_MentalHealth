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

# CONFIG
BATCH_SIZE = 200
INPUT_FILE = "beyondblue_depression_threads_all.csv"

# Read all threads
threads_df = pd.read_csv(INPUT_FILE)

# Chrome options
options = Options()
# Comment out if you want to see the browser
# options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--window-size=1920,1080")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

total_threads = len(threads_df)
batch_count = (total_threads + BATCH_SIZE - 1) // BATCH_SIZE

print(f" Total threads: {total_threads}")
print(f" Batch size: {BATCH_SIZE}")
print(f" Number of batches: {batch_count}")

for batch_num in range(batch_count):
    start_index = batch_num * BATCH_SIZE
    end_index = min(start_index + BATCH_SIZE, total_threads)
    print(f"\n Processing batch {batch_num + 1}/{batch_count}: rows {start_index}–{end_index - 1}")

    posts_data = []

    for index, row in threads_df.iloc[start_index:end_index].iterrows():
        url = row['link']
        print(f"\nProcessing [{index}] {url}")

        driver.get(url)

        try:
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.lia-message-body-content"))
            )

            soup = BeautifulSoup(driver.page_source, "html.parser")

            post_id = url.split("/")[-1]

            title_elem = soup.select_one("h1")
            title = title_elem.get_text(strip=True) if title_elem else ""

            content_elem = soup.select_one("div.lia-message-body-content")
            content = content_elem.get_text(strip=True) if content_elem else ""

            author_elem = soup.select_one("a.lia-link-navigation.lia-page-link.lia-user-name-link span")
            author = author_elem.get_text(strip=True) if author_elem else "Unknown"

            user_id_elem = soup.select_one("a.lia-link-navigation.lia-page-link.lia-user-name-link")
            user_id = ""
            if user_id_elem and "href" in user_id_elem.attrs:
                href = user_id_elem["href"]
                if "user-id/" in href:
                    user_id = href.split("user-id/")[-1]

            date_elem = soup.select_one("span.local-date")
            post_date = date_elem.get_text(strip=True) if date_elem else ""

            post_category = "Depression"

            comments = soup.select("div.lia-message-body-content")
            num_comments = len(comments) - 1

            posts_data.append({
                "Post ID": post_id,
                "Post Title": title,
                "Post Content": content,
                "Post Author": author,
                "User ID": user_id,
                "Post Date": post_date,
                "Post Category": post_category,
                "Number of Comments": num_comments,
            })

            print(f" Scraped: {title[:50]}...")

        except Exception as e:
            print(f" Skipping {url} due to error: {e}")

        time.sleep(1)

    # Save this batch
    df_batch = pd.DataFrame(posts_data)
    output_file = f"beyondblue_depression_posts_{start_index}_{end_index - 1}.csv"
    df_batch.to_csv(output_file, index=False)
    print(f" Batch saved: {output_file}")

driver.quit()

print("\n All batches completed.")
