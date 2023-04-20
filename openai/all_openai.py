import csv
import os
import time
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import random



def fetch_papers(page_number):
    url = f"https://openai.com/research?page={page_number}"
    print(f"Fetching {url}")

    # Use Chrome as the browser
    browser = webdriver.Chrome()
    browser.get(url)

    # Wait for the page to fully load
    wait = WebDriverWait(browser, 10)
    # ...
    time.sleep(random.uniform(1, 3))  # 随机等待 1 到 3 秒
    wait.until(EC.presence_of_element_located((By.ID, "research-index")))

    # Get the full HTML content
    html = browser.page_source

    # Close the browser
    browser.quit()

    # Parse the HTML content
    soup = BeautifulSoup(html, "html.parser")
    papers = []

    paper_list = soup.select("#research-index > div.container > form > div.pt-spacing-6.theme-light-gray > ul")

    # Check if the <ul> element is empty
    if not paper_list or not paper_list[0].find_all("li"):
        return []

    for paper_li in paper_list[0].find_all("li"):
        date_published = paper_li.find("span", {"class": "sr-only"}).text.strip()
        blog_name = paper_li.find("span", {"class": "f-ui-1"}).text.strip()
        blog_link = "https://openai.com" + paper_li.find("a")["href"]

        paper_link_element = paper_li.find("a", {"aria-label": "Read paper"})
        paper_link = paper_link_element["href"] if paper_link_element else None

        papers.append({
            "date_published": date_published,
            "blog_name": blog_name,
            "blog_link": blog_link,
            "paper_link": paper_link
        })
        print("date_published: ", date_published,
              "blog_name: ", blog_name,
              "blog_link: ", blog_link,
              "paper_link: ", paper_link
              )
        print('\n')

    return papers


def save_papers_to_csv(papers, filename):
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["date_published", "blog_name", "blog_link", "paper_link"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for paper in papers:
            writer.writerow(paper)


def main():
    filename = "openai_all_papers.csv"

    if os.path.exists(filename):
        print(f"File {filename} already exists, skipping fetching papers")

    all_papers = []
    max_pages = 10  # modify this to the desired maximum number of pages to fetch
    page_number = 1
    new_papers_found = True

    while new_papers_found and page_number <= max_pages:
        print(f"Fetching papers from page {page_number}...")
        papers = fetch_papers(page_number)

        if not papers:
            new_papers_found = False
        else:
            all_papers.extend(papers)
            page_number += 1
            time.sleep(1)  # add a 1-second delay between requests

    print(f"Total papers found: {len(all_papers)}")
    save_papers_to_csv(all_papers, filename)
    print(f"Papers saved to {filename}")

if __name__ == "__main__":
    main()