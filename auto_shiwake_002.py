##CSVを開く
import pandas as pd

filename = "shiwakenikkei.csv"
df = pd.read_csv(filename, encoding="Shift_JISx0213")
#読み込んだデータから使用するデータを絞り込む
columns = ["借方","借方科目名","貸方","貸方科目名","摘要"]
df_counts = df[columns].dropna()


#摘要データを単語に分ける
from janome.tokenizer import Tokenizer
t = Tokenizer()
notes = []
for ix in df_counts.index:
    note = df_counts.ix[ix,"摘要"]
    tokens = t.tokenize(note.replace('　',' '))
    words = ""
    for token in tokens:
        words += " " + token.surface
    notes.append(words.replace(' \u3000', ''))
#トークンのベクトル化
from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer()
vect.fit(notes) #トレーニングデータの統計値をセット
X = vect.transform(notes) #トレーニングデータで書き換え→ベクトル化

##借方
#教師データとして勘定科目のコードを使用
y = df_counts.借方.as_matrix().astype("int").flatten()
#数値に変換したデータを交差検証（クロスバリデーション）で学習データと検証データに分割
from sklearn import cross_validation
test_size = 0.2
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=test_size)
#分割したデータを使って学習
from sklearn.svm import LinearSVC
clf = LinearSVC(C=120.0, random_state=42)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)


##貸方
#教師データとして勘定科目のコードを使用
yc = df_counts.貸方.as_matrix().astype("int").flatten()
#数値に変換したデータをクロスバリデーションで学習データと検証データに分割
from sklearn import cross_validation
test_size = 0.2
X_train, X_test, yc_train, yc_test = cross_validation.train_test_split(X, yc, test_size=test_size)
#分割したデータを使って学習
from sklearn.svm import LinearSVC
clf_c = LinearSVC(C=120.0, random_state=42)
clf_c.fit(X_train, yc_train) #トレーニングデータで統計値セット
clf_c.score(X_test, yc_test) #テストデータでスコア算出



#勘定科目の予測：テストデータ入力
tests = [
    "パン",
    "野菜",
    "マット代",
    "羊のむくむく",
    "喜多野　野菜",
    "アルバイト",
    "電気代",
    "物販小売",
    "高山産業",
    "預入れ",
    "ひまわり会計事務所",
    "蒲原"
    ]   

#テストデータの形態素解析→BOW→ベクトル化
notes = []
for note in tests:
    tokens = t.tokenize(note)
    words = ""
    for token in tokens:
        words += " " + token.surface
    notes.append(words)

X = vect.transform(notes)

#借方予測→コードから科目名へ
result_d = clf.predict(X)
df_rs = df_counts[["借方科目名", "借方"]]
df_rs.index = df_counts["借方"].astype("int")
df_rs = df_rs[~df_rs.index.duplicated()]["借方科目名"]


#貸方予測→コードから科目名へ
result_c = clf_c.predict(X)
df_rsc = df_counts[["貸方科目名", "貸方"]]
df_rsc.index = df_counts["貸方"].astype("int")
df_rsc = df_rsc[~df_rsc.index.duplicated()]["貸方科目名"]


for i in range(len(tests)):
    print(tests[i], result_d[i], df_rs.ix[result_d[i]],result_c[i], df_rsc.ix[result_c[i]])

