import pandas as pd

def concat_item_metadata(row):
    meta = ""
    flag = False
    if row['title'] is not None:
        meta += row['title']
        flag = True
    if len(row['features']) > 0:
        if flag:
            meta += ' '
        meta += ' '.join(row['features'])
        flag = True
    if len(row['description']) > 0:
        if flag:
            meta += ' '
        meta += ' '.join(row['description'])
    cleaned_meta = (meta
                    .replace('\t', ' ')
                    .replace('\n', ' ')
                    .replace('\r', '')
                    .strip()
                    )
    return cleaned_meta


def filter_reviews(row, valid_timestamp, all_cleaned_item_metadata):
    # Downsampling
    # pr = random.randint(1, downsampling_factor)
    # if pr > 1:
    #     return False
    if row['timestamp'] >= valid_timestamp:
        return False
    asin = row['parent_asin']
    if asin not in all_cleaned_item_metadata:
        return False
    if len(row['review']) <= 30:
        return False
    return True


def concat_user_review(row):
    review = ""
    flag = False
    if row['title'] is not None:
        review += row['title']
        flag = True
    if row['text'] is not None:
        if flag:
            review += ' '
        review += row['text']
    cleaned_review = (review
                      .replace('\t', ' ')
                      .replace('\n', ' ')
                      .replace('\r', '')
                      .strip()
                      )    
    return cleaned_review

def prep_user_nodes(user_df):
    df = user_df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    max_date = df['timestamp'].max()
    df['last_active_in_days'] = (max_date - df['timestamp']).dt.days
    df['review'] = df['title'] + " " + df['text']
    df['review'] = (df['review']
                    .replace('\t', ' ')
                    .replace('\n', ' ')
                    .replace('\r', '')
                    .str
                    .strip()
    )
    df['word_count'] = df['review'].str.split().str.len()
    return df

def make_user_nodes(user_df):
    df = user_df.copy()
    df = (df
          .groupby('user_id')
          .agg({
              'rating': ['mean', 'count'],
              'helpful_vote': ['mean', lambda x: (x >= 1).sum()], 
              'verified_purchase': 'mean', 
              'last_active_in_days': ['min', 'max'], 
              'word_count': 'mean', 
              'review': lambda x: ' || '.join(x)
              })
    )
    df.columns = [
        'rating_mean', 'rating_count',
        'helpful_vote_mean', 'helpful_vote_gte_1',
        'verified_purchase_mean',
        'last_active_in_days_min', 'last_active_in_days_max',
        'word_count_mean',
        'reviews'
    ]
    df = df.reset_index()
    return df