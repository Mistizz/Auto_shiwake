import pandas as pd
from janome.tokenizer import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score  # 交差検証用
from sklearn.svm import LinearSVC
import os

# デスクトップのパスを取得
#desktop_path = os.getenv("HOMEDRIVE") + os.getenv("HOMEPATH") + "\\Desktop\\"
#output_folder = "shiwake\\"

## CSVを開く
filename_train_csv = "nikkeihyou.csv"  # trainデータ名

#path_train_csv = desktop_path + output_folder + filename_train_csv
path_train_csv = filename_train_csv

df = pd.read_csv(path_train_csv, encoding="Shift_JISx0213")
# 読み込んだデータから使用するデータを絞り込む
columns = ["借方科目コード", "借方科目名", "貸方科目コード", "貸方科目名", "摘要"]
df_counts = df[columns].dropna()

# 摘要データを単語に分ける
t = Tokenizer()
notes = []
for ix in df_counts.index:
    note = df_counts.loc[ix, "摘要"]  # ix から loc へ
    tokens = t.tokenize(note.replace('　', ' '))
    words = ""
    for token in tokens:
        words += " " + token.surface
    notes.append(words.replace(' \u3000', ''))

# トークンのベクトル化
vect = TfidfVectorizer()
vect.fit(notes)  # トレーニングデータの統計値をセット
X = vect.transform(notes)  # トレーニングデータで書き換え→ベクトル化

## 借方科目名のトークン化
td = Tokenizer()
dnotes = []
for ix in df_counts.index:
    dnote = df_counts.loc[ix, "借方科目名"]  # ix から loc へ
    dtokens = td.tokenize(dnote.replace('　', ' '))
    dwords = ""
    for dtoken in dtokens:
        dwords += " " + dtoken.surface
    dnotes.append(dwords.replace(' \u3000', ''))
yd = dnotes

## 貸方科目名のトークン化
tc = Tokenizer()
cnotes = []
for ix in df_counts.index:
    cnote = df_counts.loc[ix, "貸方科目名"]  # ix から loc へ
    ctokens = tc.tokenize(cnote.replace('　', ' '))
    cwords = ""
    for ctoken in ctokens:
        cwords += " " + ctoken.surface
    cnotes.append(cwords.replace(' \u3000', ''))
yc = cnotes

# 借方科目の交差検証
dclf = LinearSVC(C=120.0, random_state=42)
scores_d = cross_val_score(dclf, X, yd, cv=5)  # cv=5 で5分割交差検証を実行
print("借方科目の交差検証スコア:", scores_d)
print("借方科目の平均スコア:", scores_d.mean())

# 貸方科目の交差検証
cclf = LinearSVC(C=120.0, random_state=42)
scores_c = cross_val_score(cclf, X, yc, cv=5)  # cv=5 で5分割交差検証を実行
print("貸方科目の交差検証スコア:", scores_c)
print("貸方科目の平均スコア:", scores_c.mean())

## testdata.csvを開く
filename_test_csv = "input_data.csv"
#path_test_csv = desktop_path + output_folder + filename_test_csv
path_test_csv = filename_test_csv
df_test = pd.read_csv(path_test_csv, encoding="utf-8")
# 読み込んだデータから使用するデータを絞り込む
df_tekiyou = df_test.iloc[:, 16]

# 勘定科目の予測：テストデータ入力
tests = df_tekiyou

# テストデータの形態素解析→BOW→ベクトル化
notes_t = []
for note_t in tests:
    tokens_t = t.tokenize(note_t)
    words_t = ""
    for token_t in tokens_t:
        words_t += " " + token_t.surface
    notes_t.append(words_t)

X_test = vect.transform(notes_t)

# 借方予測→コードから科目名へ
dclf.fit(X, yd)  # 学習データ全体を使ってモデルを再学習
result_d = dclf.predict(X_test)
df_rs = df_counts[["借方科目名", "借方科目コード"]]
df_rs.index = df_counts["借方科目コード"].astype("int")
df_rs = df_rs[~df_rs.index.duplicated()]["借方科目名"]

# 貸方予測→コードから科目名へ
cclf.fit(X, yc)  # 学習データ全体を使ってモデルを再学習
result_c = cclf.predict(X_test)
df_rsc = df_counts[["貸方科目名", "貸方科目コード"]]
df_rsc.index = df_counts["貸方科目コード"].astype("int")
df_rsc = df_rsc[~df_rsc.index.duplicated()]["貸方科目名"]

## 仕訳出力
# testdataのデータフレームに結果を出力
df_test.iloc[:, 5] = [i.replace(" ", "") for i in result_d]
df_test.iloc[:, 11] = [i.replace(" ", "") for i in result_c]

# デスクトップの[shiwake]フォルダにCSVで出力
csv_name = "YA05-SWK_result.csv"
#path_output = desktop_path + output_folder + csv_name
path_output = csv_name


df_test.to_csv(path_output, header=False, index=False, encoding="Shift_JISx0213")

print(path_output)
print("出力が完了しました。")
