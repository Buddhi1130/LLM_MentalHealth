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

# Updated forum URL
BASE_URL = "https://forums.beyondblue.org.au/t5/suicidal-thoughts-and-self-harm/bd-p/c1-sc2-b4"

options = Options()
# options.add_argument("--headless")  # Optional: run headless
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--window-size=1920,1080")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

threads = []

driver.get(BASE_URL)
page_counter = 1

while True:
    print(f"Processing page {page_counter}...")

    try:
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "a[title]"))
        )
    except Exception as e:
        print(f" Could not find threads on page {page_counter}. Stopping. Reason: {e}")
        break

    soup = BeautifulSoup(driver.page_source, "html.parser")
    thread_links = soup.select("a[title]")

    for link_element in thread_links:
        href = link_element.get("href", "")
        if href.startswith("/t5/suicidal-thoughts-and-self-harm/"):
            title = link_element.get_text(strip=True)
            link = "https://forums.beyondblue.org.au" + href
            threads.append({"title": title, "link": link})

    # Check for "Next Page"
    next_link_elem = soup.select_one('a[aria-label="Next Page"]')
    if next_link_elem:
        next_href = next_link_elem.get("href")
        if next_href:
            next_url = (
                next_href if next_href.startswith("http")
                else "https://forums.beyondblue.org.au" + next_href
            )
            print(f" Going to: {next_url}")
            driver.get(next_url)
            page_counter += 1
            time.sleep(2)
            continue

    print("Scraping complete.")
    break

driver.quit()

# Save to CSV
df = pd.DataFrame(threads).drop_duplicates()
df.to_csv("beyondblue_suicidal_threads_all.csv", index=False)

print(f" Scraped {len(df)} threads successfully.")
