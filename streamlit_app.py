import streamlit as st

"""\
# Message Analysis

Analyzing text messages with NLP techniques and Streamlit

## Fetching messages database from MacOS

You'll need to access `~/Library/Messages/chat.db` either by allowing terminal access in settings.
Then you can copy it to your current folder with `cp ~/Library/Messages/chat.db ./chat.db`
"""
with st.expander("Show Image"):
    st.image("images/security.png")
"""\
or use Finder's 'Go' drop down from the top menu and hold option / alt.
Select Library then the Messages folder then manually copy `chat.db`

## What Tables are available
"""

with st.echo():
    import sqlite3

    conn = sqlite3.connect("chat.db")
    cursor = conn.cursor()
    cursor.execute(
        """\
    SELECT name, sql FROM sqlite_schema
    WHERE type='table'
    ORDER BY name;"""
    )
    for name, sql in cursor.fetchall():
        with st.expander(name):
            st.code(sql.replace(",", ",\n"))

"""\
## Message Partners and Handles

The `message` and `handle` tables are the most interesting for this analysis.

We'll use Pandas and Plotly here to explore some of the data.
Using sqlite3 would work just fine, or even a GUI such as dbeaver or DB Browser for SQLite
"""

with st.echo():
    import pandas as pd

    message = pd.read_sql("SELECT * FROM message", conn)
    message

"""\
Let's check out just the interesting columns
"""

with st.echo():
    message = pd.read_sql("SELECT text, is_from_me, handle_id, date FROM message", conn)
    message


"""\
We'll need to update the date timestamps to understand them as humans

(from [2017 SO](https://apple.stackexchange.com/questions/114168/dates-format-in-messages-chat-db))
"""

with st.echo():
    message = pd.read_sql(
        """SELECT text, is_from_me, handle_id,
        datetime(message.date/1000000000 + strftime("%s", "2001-01-01") ,"unixepoch","localtime") as 'date'
        FROM message""",
        conn,
    )
    message

"""\
We can also get the phone number instead of just the number assigned to each contact by joining with the `handle` table
"""

with st.echo():
    message = pd.read_sql(
        """SELECT text, is_from_me, h.id,
        datetime(m.date/1000000000 + strftime("%s", "2001-01-01") ,"unixepoch","localtime") as 'date'
        FROM message m JOIN handle h ON m.handle_id == h.rowid""",
        conn,
    )
    message

"""\
## Messages Per Partner

From here we can dig into the trends.
Lets count how many texts were sent between each person.
"""

with st.echo():
    message_counts = message.id.value_counts()
    message_counts

"""\
## Message Sentiment

Now lets get sentimental.
With Natural Language Toolkit we can assess how positive or negative each message is.

Here's the basic breakdown
"""

with st.echo():
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer

    @st.experimental_singleton
    def get_analyzer():
        nltk.download("vader_lexicon")
        analyzer = SentimentIntensityAnalyzer()
        return analyzer

    analyzer = get_analyzer()
    text_input = st.text_area(
        "Enter text to analyze", "Wow, NLTK + Streamlit is really powerful!"
    )
    result = analyzer.polarity_scores(text_input)
    st.write(result)

"""\
The first 3 numbers represent how negative, positive, or neutral the model thinks the text is.
These scores add up to 1 and are percentages of "how much of the text is positive".

The compound score is normalized and weighted against an English lexicon.
These range from -1 to 1, with 1 being the most positve and -1 being the most negative.

We'll add all the scores to our data for now, [read the docs](https://github.com/cjhutto/vaderSentiment#about-the-scoring) for more.
"""
with st.echo():

    def add_scores(df: pd.DataFrame):
        raw_text = ""
        if not pd.isna(df.text):
            raw_text = df.text
        result = analyzer.polarity_scores(raw_text)
        df["negative"] = result["neg"]
        df["neutral"] = result["neu"]
        df["positive"] = result["pos"]
        df["compound"] = result["compound"]
        return df

    scored_message = message.apply(add_scores, axis=1)
    scored_message

"""\
From here the world is your oyster!

Want to see if you send happier texts on weekends?
Group texts by weekday then sum or average the sentiments per day!

Want to see who you have the rockiest relationship with?
Group texts by id and by whom it was sent.
Compare the biggest gap between the sender and receiver!

Want to see if your texting got more positive over time?
Group texts by timestamp and chart the scores over time!

## Most Extreme Messages

The most positive and negative messages sent and received
"""

with st.echo():
    sent_messages = scored_message.loc[scored_message.is_from_me == 1].reset_index()

    most_positive_sent = sent_messages.iloc[sent_messages.compound.idxmax()]
    least_positive_sent = sent_messages.iloc[sent_messages.compound.idxmin()]

    st.subheader("Most Positive Message I sent")
    st.write(most_positive_sent.astype(str))
    st.subheader("Most Negative Message I sent")
    st.write(least_positive_sent.astype(str))

    received_messages = scored_message.loc[scored_message.is_from_me == 0].reset_index()

    most_positive_received = received_messages.iloc[received_messages.compound.idxmax()]
    least_positive_received = received_messages.iloc[
        received_messages.compound.idxmin()
    ]

    st.subheader("Most Negative Message I received")
    st.write(least_positive_received.astype(str))
    st.subheader("Most Positive Message I received")
    st.write(most_positive_received.astype(str))

"""\
## Average Compound Per Partner
"""

with st.echo():
    compound_scores = scored_message[['id', 'compound']].groupby('id').mean()
    st.bar_chart(compound_scores)