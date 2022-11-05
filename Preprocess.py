import Dataset

df_train = get_finance_train()
df_test = get_finance_test()

df_train['Sentence'] = df_train['Sentence'].apply(clean_text)
df_test['Sentence'] = df_test['Sentence'].apply(clean_text)

MAX_SEQUENCE_LENGTH = 256
MAX_NB_WORDS = 1000

X_train = pad_sequences_train(df_train, df_test)
X_test = pad_sequences_test(df_train, df_test)

y_train = pd.get_dummies(df_train['Label']).values
y_test = pd.get_dummies(df_test['Label']).values

