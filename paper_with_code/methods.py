import requests
from bs4 import BeautifulSoup

import requests
from bs4 import BeautifulSoup
URL = "https://paperswithcode.com"
url = "https://paperswithcode.com/methods"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")


categories = soup.select("div.infinite-container.featured-methods h4 a")
print(categories)
for category in categories:
    category_title = category.text.strip()
    category_url = URL + category["href"]
    print(f"Category: {category_title}")
    print(f"URL: {category_url}\n")

    # make a request to the category page
    response = requests.get(category_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # select all methods in the category using the CSS selector
    methods = [method for method in soup.select("div.infinite-container.featured-methods h2 a")]
    for method in methods:
        method_title = method.text.strip()
        # get the method URL
        method_url = URL + method['href']
        print(f"Method: {method_title}")
        print(f"URL: {method_url}\n")

        # make a request to the method page
        response = requests.get(method_url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # extract the required information from the method page
        method_title = soup.select_one('h1').text
        method_score = soup.select_one('span.score').text
        method_description = soup.select_one('div.panel-body').text.strip()

        # print the information for the method
        print(f"Title: {method_title}\nScore: {method_score}\nDescription: {method_description}\n")
