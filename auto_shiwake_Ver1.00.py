import pandas as pd
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cross_validation
from sklearn.svm import LinearSVC

##CSVを開く
filename = "nikkei.csv"
df = pd.read_csv(filename, encoding="Shift_JISx0213")
#読み込んだデータから使用するデータを絞り込む
columns = ["借方","借方科目名","貸方","貸方科目名","摘要"]
df_counts = df[columns].dropna()


#摘要データを単語に分ける
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
vect = TfidfVectorizer()
vect.fit(notes) #トレーニングデータの統計値をセット
X = vect.transform(notes) #トレーニングデータで書き換え→ベクトル化


##借方科目名のトークン化
td = Tokenizer()
dnotes = []
for ix in df_counts.index:
    dnote = df_counts.ix[ix,"借方科目名"]
    dtokens = td.tokenize(dnote.replace('　',' '))
    dwords = ""
    for dtoken in dtokens:
        dwords += " " + dtoken.surface
    dnotes.append(dwords.replace(' \u3000', ''))
yd = dnotes

##貸方科目名のトークン化
tc = Tokenizer()
cnotes = []
for ix in df_counts.index:
    cnote = df_counts.ix[ix,"貸方科目名"]
    ctokens = tc.tokenize(cnote.replace('　',' '))
    cwords = ""
    for ctoken in ctokens:
        cwords += " " + ctoken.surface
    cnotes.append(cwords.replace(' \u3000', ''))
yc = cnotes

##借方
#数値に変換したデータを交差検証（クロスバリデーション）で学習データと検証データに分割
test_size = 0.2
X_train, X_test, yd_train, yd_test = cross_validation.train_test_split(X, yd, test_size=test_size)
#分割したデータを使って学習
dclf = LinearSVC(C=120.0, random_state=42)
dclf.fit(X_train, yd_train)
dr = dclf.score(X_test, yd_test)


##貸方
#数値に変換したデータを交差検証（クロスバリデーション）で学習データと検証データに分割
test_size = 0.2
X_train, X_test, yc_train, yc_test = cross_validation.train_test_split(X, yc, test_size=test_size)
#分割したデータを使って学習
cclf = LinearSVC(C=120.0, random_state=42)
cclf.fit(X_train, yc_train)
cr = cclf.score(X_test, yc_test)

##testdata.csvを開く
filename = "input_data.csv"
df_test = pd.read_csv(filename, encoding="Shift_JISx0213")
#読み込んだデータから使用するデータを絞り込む
df_tekiyou = df_test.iloc[:, 16]



#勘定科目の予測：テストデータ入力
tests = df_tekiyou

     
#テストデータの形態素解析→BOW→ベクトル化
notes_t = []
for note_t in tests:
    tokens_t = t.tokenize(note_t)
    words_t = ""
    for token_t in tokens_t:
        words_t += " " + token_t.surface
    notes_t.append(words_t)

X = vect.transform(notes_t)

#借方予測→コードから科目名へ
result_d = dclf.predict(X)
df_rs = df_counts[["借方科目名", "借方"]]
df_rs.index = df_counts["借方"].astype("int")
df_rs = df_rs[~df_rs.index.duplicated()]["借方科目名"]

#貸方予測→コードから科目名へ
result_c = cclf.predict(X)
df_rsc = df_counts[["貸方科目名", "貸方"]]
df_rsc.index = df_counts["貸方"].astype("int")
df_rsc = df_rsc[~df_rsc.index.duplicated()]["貸方科目名"]

##仕訳出力
#for i in range(len(tests)):
#    print(tests[i], result_d[i], df_rs.ix[result_d[i]], result_c[i], df_rsc.ix[result_c[i]])

#testdataのデータフレームに結果を出力
df_test.iloc[:, 5] = [i.replace(" ","") for i in result_d]
df_test.iloc[:, 11] = [i.replace(" ","") for i in result_c]

#print(df_test)

#デスクトップにCSVで出力
df_test.to_csv(r"C:\Users\mjsadmin\Desktop\YA05-SWK_result.csv", header=False, index=False, encoding="Shift_JISx0213")

print("出力が完了しました。")
