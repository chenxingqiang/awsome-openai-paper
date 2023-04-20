
ALL_AUTHORS_URLNAME = [
    'martin-abadi','pieter-abbeel',
    'joshua-achiam','steven-adler',
    'sandhini-agarwal','ilge-akkaya',
    'maruan-al-shedivat','dario-amodei',
    'daniella-amodei','daniela-amodei',
    'marcin-andrychowicz','tasmin-asfour',
    'amanda-askell','anish-athalye',
    'igor-babuschkin','bowen-baker',
    'suchir-balaji','trapit-bansal',
    'yamini-bansal','boaz-barak',
    'elizabeth-barnes','ben-barry',
    'peter-l-bartlett','mohammad-bavarian',
    'alexandre-m-bayen','christopher-berner',
    'jesse-bettencourt','lukas-biewald',
    'xue-bin-peng','trevor-blackwell',
    'greg-brockman','tom-brown',
    'miles-brundage','yura-burda',
    'nick-cammarata','andrew-n-carr',
    'shan-carter','brooke-chan',
    'fotios-chantzis','peter-chen',
    'richard-chen','xi-chen',
    'mark-chen','ricky-t-q-chen',
    'benjamin-chess','vicki-cheung',
    'rewon-child','maciek-chociej',
    'paul-christiano','casey-chu',
    'jack-clark','jeff-clune',
    'karl-cobbe','taco-cohen',
    'dave-cummings','andrew-m-dai',
    'trevor-darrell','przemyslaw-debiak',
    'akshay-degwekar','christy-dennison',
    'filip-de-turck','prafulla-dhariwal',
    'yilun-du','yan-duan',
    'david-duvenaud','harri-edwards',
    'alexei-a-efros','tyna-eloundou',
    'ulfar-erlingsson','owain-evans',
    'david-farhi','chelsea-finn',
    'quirin-fischer','carlos-florensa',
    'jakob-foerster','rachel-fong',
    'davis-foote','kevin-frans',
    'deep-ganguli','leo-gao',
    'jon-gauthier','gabriel-goh',
    'ian-goodfellow','jonathan-gordon',
    'will-grathwohl','scott-gray',
    'roger-grosse','aditya-grover',
    'jayesh-k-gupta','william-guss',
    'chris-hallacy','jesse-michael-han',
    'ankur-handa','jean-harb',
    'shariq-hashme','johannes-heidecke',
    'dan-hendrycks','tom-henighan',
    'ariel-herbert-voss','danny-hernandez',
    'christopher-hesse','jacob-hilton',
    'jonathan-ho','rein-houthooft',
    'kenny-hsu','sandy-huang',
    'daniel-huang','joost-huizinga',
    'geoffrey-irving','phillip-isola',
    'shantanu-jain','shawn-jain',
    'joanne-jang','nicholas-joseph',
    'rafal-jozefowicz','heewoo-jun',
    'lukasz-kaiser','sham-kakade',
    'daniel-kang','ingmar-kanitscheider',
    'jared-kaplan','gal-kaplun',
    'andrej-karpathy','tabarak-khan',
    'heidy-khlaaf','jong-wook-kim',
    'durk-kingma','oleg-klimov',
    'matthew-knight','vineet-kosaraju',
    'gretchen-krueger','vikash-kumar','david-lansky',
    'joel-lehman','jan-leike',
    'sergey-levine','shun-liao',
    'stephanie-lin','mateusz-litwin',
    'christos-louizos','ryan-lowe',
    'kendall-lowrey','david-luan',
    'benjamin-mann','todor-markov',
    'tambet-matiisen','katie-mayer',
    'sam-mccandlish','bob-mcgrew',
    'christine-mcleavey-payne','dimitris-metaxas',
    'smitha-milli','pamela-mishkin',
    'nikhil-mishra','vedant-misra',
    'takeru-miyato','igor-mordatch',
    'evan-morikawa','mira-murati',
    'reiichiro-nakano','preetum-nakkiran',
    'kamal-ndousse','arvind-neelakantan',
    'alex-nichol','chris-olah',
    'avital-oliver','catherine-olsson',
    'long-ouyang','jakub-pachocki',
    'michael-page','alex-paino',
    'nicolas-papernot','deepak-pathak',
    'mikhail-pavlov','arthur-petron',
    'michael-petrov','vicki-pfau',
    'lerrel-pinto','matthias-plappert',
    'stanislas-polu','henrique-ponde',
    'glenn-powell','boris-power',
    'alethea-power','eric-price',
    'raul-puri','alec-radford',
    'jonathan-raiman','aravind-rajeswaran',
    'aditya-ramesh','alex-ray','erika-reinhardt',
    'raphael-ribas','nick-ryder',
    'ruslan-salakhutdinov','tim-salimans',
    'raul-sampedro','girish-sastry',
    'william-saunders','larissa-schiavo',
    'jeremy-schlatter','jonas-schneider',
    'david-schnurr','ludwig-schubert',
    'john-schulman-2','zain-shah','toki-sherbakov','pranav-shyam',
    'szymon-sidor','eric-sigler',
    'irene-solaiman','dawn-song','bradly-stadie',
    'kenneth-o-stanley','jacob-steinhardt',
    'nisan-stiennon','adam-stooke',
    'amos-storkey','joseph-suarez',
    'melanie-subbiah','felipe-petroski-such',
    'yi-sun','ilya-sutskever',
    'kunal-talwar','aviv-tamar',
    'alex-tamkin','jie-tang',
    'haoran-tang','nikolas-tezak',
    'madeleine-thompson','philippe-tillet',
    'josh-tobin','emanuel-todorov',
    'jerry-tworek','vinod-vaikuntanathan',
    'chelsea-voss','justin-jay-wang',
    'peter-welinder','max-welling',
    'lilian-weng','shimon-whiteson',
    'clemens-winter','filip-wolski',
    'yi-wu','yuhuai-wu',
    'jeffrey-wu','cathy-wu',
    'tao-xu','tristan-yang',
    'ge-yang','catherine-yeh',
    'cathy-yeh','diane-yoon',
    'qiming-yuan','wojciech-zaremba',
    'susan-zhang','lei-zhang',
    'han-zhang','peter-zhokhov',
    'daniel-ziegler','openai']


import csv
import os
import time
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import random



def fetch_papers(author):
    url = f"https://openai.com/research?authors={author}"
    print(f"Fetching {url}")

    # Use Chrome as the browser
    browser = webdriver.Chrome()
    browser.get(url)

    # Wait for the page to fully load
    wait = WebDriverWait(browser, 10)
    # ...
    time.sleep(random.uniform(5, 10))  # 随机等待 1 到 3 秒
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
            "author": author,
            "blog_name": blog_name,
            "blog_link": blog_link,
            "paper_link": paper_link
        })
        print("date_published: ", date_published,
              "author:", author,
              "blog_name: ", blog_name,
              "blog_link: ", blog_link,
              "paper_link: ", paper_link
              )
        print('\n')

    return papers


def save_papers_to_csv(papers, filename):
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["date_published",'author', "blog_name", "blog_link", "paper_link"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for paper in papers:
            writer.writerow(paper)


def main():
    filename = "openai_author_papers.csv"

    if os.path.exists(filename):
        print(f"File {filename} already exists, skipping fetching papers")

    all_papers = []
    author = 1
    new_papers_found = True

    for author in ALL_AUTHORS_URLNAME:
        print(f"Fetching papers from  {author}...")
        papers = fetch_papers(author)

        if not papers:
            new_papers_found = False
        else:
            all_papers.extend(papers)

            time.sleep(random.uniform(1,3))  # add a 1-second delay between requests

    print(f"Total papers found: {len(all_papers)}")
    save_papers_to_csv(all_papers, filename)
    print(f"Papers saved to {filename}")

if __name__ == "__main__":
    main()
