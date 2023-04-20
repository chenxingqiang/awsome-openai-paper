import csv
import requests

# Path to the CSV file containing the paper data
csv_path = "/Users/xingqiangchen/PyProjects/search_papers/openai/data/openai_all_papers.csv"

# Loop through each row in the CSV file
with open(csv_path) as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        try:
            if 'abs' in row["paper_link"]:
                link = row["paper_link"].replace('abs', 'pdf') + '.pdf'
            else:
                link = row["paper_link"]
            response = requests.get(link)
            # Extract the date_published and blog_name from the row
            date_published = row["date_published"].replace(',','-').replace(" ", "")
            blog_name = row["blog_name"]
            # Use the date_published and blog_name to determine the filename
            filename = f"./data/papers/{date_published}_{blog_name}.pdf"
            with open(filename, "wb") as outfile:
                outfile.write(response.content)
        except:
            print(f"Failed to download paper {row['paper_link']}")