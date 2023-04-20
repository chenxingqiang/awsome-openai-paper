import pandas as pd
import matplotlib.pyplot as plt

# Read in the data from the CSV file
data_path = "/Users/xingqiangchen/PyProjects/search_papers/openai/data/openai_all_papers.csv"
df = pd.read_csv(data_path)

# Extract the year from the "date_published" column
df['year'] = pd.to_datetime(df['date_published']).dt.year

# Plot the number of papers published each year
year_counts = df['year'].value_counts().sort_index()
plt.plot(year_counts.index, year_counts.values)
plt.xlabel('Year')
plt.ylabel('Number of Papers Published')
plt.savefig('openai_paper_line_year.png') # saving the plot as png file
plt.title('OpenAI Papers Published by Year')
plt.show()
plt.close()


# Read in the data from the CSV file
data_path = "/Users/xingqiangchen/PyProjects/search_papers/openai/data/openai_all_papers.csv"
df = pd.read_csv(data_path)

# Extract the year from the "date_published" column
df['year'] = pd.to_datetime(df['date_published']).dt.year

# Plot the histogram of the number of papers published each year
plt.hist(df['year'], edgecolor='black', bins=range(min(df['year']), max(df['year']) + 1))
plt.xlabel('Year')
plt.ylabel('Number of Papers Published')
plt.title('Histogram of OpenAI Papers Published by Year')
plt.savefig('openai_paper_histogram_year.png') # saving the plot as png file
plt.show()
plt.close()


# Read in the data from the CSV file
data_path = "/Users/xingqiangchen/PyProjects/search_papers/openai/data/openai_all_papers.csv"
df = pd.read_csv(data_path)

# Convert the "date_published" column to Pandas datetime format
df['date_published'] = pd.to_datetime(df['date_published'])

# Extract the month from the "date_published" column
df['month_published'] = df['date_published'].dt.to_period('M')

# Count the number of papers published each month and convert the Periods to Timestamps
papers_per_month = df['month_published'].value_counts().sort_index()
papers_per_month = pd.Series(papers_per_month.values, index=papers_per_month.index.to_timestamp())

# Plot the number of papers published each month
plt.plot(papers_per_month.index, papers_per_month.values, color='dodgerblue')
plt.xlabel('Month')
plt.ylabel('Number of Papers Published')
plt.title('Number of OpenAI Papers Published per Month')

# Customize the x-axis labels to show only every 6th month
tick_labels = papers_per_month.index.astype(str)
tick_labels = tick_labels[::6]
plt.xticks(tick_labels, rotation=45)

# Add a grid to the plot
plt.grid(alpha=0.3)

plt.savefig('openai_paper_line_month.png')

plt.show()




# Read in the data from the CSV file
data_path = "/Users/xingqiangchen/PyProjects/search_papers/openai/data/openai_author_papers.csv"
df = pd.read_csv(data_path)

# Count the number of papers per author
papers_per_author = df['author'].value_counts()

# extract the top 20 authors by paper count
top_authors = papers_per_author.nlargest(20)

# plot the pie chart for the top 20 authors
plt.pie(top_authors, labels=top_authors.index, autopct='%1.1f%%')
plt.title('Distribution of Papers Among Top 20 Authors')
plt.savefig('openai_paper_pie_authortop20.png') # saving the plot as png
plt.show()



import pandas as pd
import matplotlib.pyplot as plt

# Read in the data from the CSV file
# data_path = "/Users/xingqiangchen/PyProjects/search_papers/openai/data/openai_author_papers.csv"
# df = pd.read_csv(data_path)

# # Extract the year from the 'date' column
# df['year'] = pd.DatetimeIndex(df['date_published']).year
# # extract the counts of papers by author
# # extract the counts of papers by author
# papers_per_author = df['author'].value_counts()

# # extract the top 20 authors by paper count
# top_authors = papers_per_author.nlargest(20).tolist()

# # extract the counts of papers by author and year
# papers_per_author_per_year = df.groupby(['author', 'year']).size().unstack()

# # only keep data for the top authors
# papers_per_author_per_year = papers_per_author_per_year[top_authors]

# # plot the line chart for the top authors' paper counts
# papers_per_author_per_year.plot(kind='line', marker='o')
# plt.title('Number of Papers per Year for Top 20 Authors')
# plt.xlabel('Year')
# plt.ylabel('Number of Papers')
# plt.savefig('openai_paper_bar_author_year.png') # saving the plot as png
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# load the data into dataframes
path = "/Users/xingqiangchen/PyProjects/search_papers/openai/data"
all_papers = pd.read_csv(path + '/openai_all_papers.csv')
author_papers = pd.read_csv(path +'/openai_author_papers.csv')
model_papers = pd.read_csv(path+'/openai_model_papers.csv')
topic_papers = pd.read_csv(path+'/openai_topic_papers.csv')
types_papers = pd.read_csv(path+'/openai_types_papers.csv')

# extract the counts of papers by date for each dataframe
all_counts = all_papers['date_published'].value_counts().sort_index()
author_counts = author_papers['date_published'].value_counts().sort_index()
model_counts = model_papers['date_published'].value_counts().sort_index()
topic_counts = topic_papers['date_published'].value_counts().sort_index()
types_counts = types_papers['date_published'].value_counts().sort_index()

# plot the counts
plt.plot(all_counts.index, all_counts.values, label='All Papers')
plt.plot(author_counts.index, author_counts.values, label='By Author')
plt.plot(model_counts.index, model_counts.values, label='By Model')
plt.plot(topic_counts.index, topic_counts.values, label='By Topic')
plt.plot(types_counts.index, types_counts.values, label='By Type')
plt.legend()
plt.xlabel('Date Published')
plt.ylabel('Number of Papers')
plt.title('OpenAI Papers Published Over Time')
plt.show()

# plot the counts by author
author_grouped = author_papers.groupby('author')['date_published'].value_counts().sort_values(ascending=False).reset_index(name='count')
top_authors = author_grouped['author'].value_counts().head(10).index
for author in top_authors:
    author_subset = author_grouped[author_grouped['author'] == author]
    plt.plot(author_subset['date_published'], author_subset['count'], label=author)
plt.legend()
plt.xlabel('Date Published')
plt.ylabel('Number of Papers')
plt.title('Top Authors of OpenAI Papers')
plt.show()

# plot the counts by model
model_grouped = model_papers.groupby('model')['date_published'].value_counts().sort_values(ascending=False).reset_index(name='count')
top_models = model_grouped['model'].value_counts().head(10).index
for model in top_models:
    model_subset = model_grouped[model_grouped['model'] == model]
    plt.plot(model_subset['date_published'], model_subset['count'], label=model)
plt.legend()
plt.xlabel('Date Published')
plt.ylabel('Number of Papers')
plt.title('Top Models in OpenAI Papers')
plt.show()

# plot the counts by topic
topic_grouped = topic_papers.groupby('topic')['date_published'].value_counts().sort_values(ascending=False).reset_index(name='count')
top_topics = topic_grouped['topic'].value_counts().head(10).index
for topic in top_topics:
    topic_subset = topic_grouped[topic_grouped['topic'] == topic]
    plt.plot(topic_subset['date_published'], topic_subset['count'], label=topic)
plt.legend()
plt.xlabel('Date Published')
plt.ylabel('Number of Papers')
plt.title('Top Topics in OpenAI Papers')
plt.show()

# plot the counts by paper type
types_grouped = types_papers.groupby('types')['date_published'].value_counts().sort_values(ascending=False).reset_index(name='count')
for types in types_grouped['types'].unique():
    types_subset = types_grouped[types_grouped['types'] == types]
    plt.plot(types_subset['date_published'], types_subset['count'], label=types)
plt.legend()
plt.xlabel('Date Published')
plt.ylabel('Number of Papers')
plt.title('OpenAI Papers by Type')
plt.show()
