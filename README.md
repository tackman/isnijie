# isnijie
入力画像が二次元イラストかどうかのニューラルネットワーク実装です。

# 必要環境
- Keras
- TensorFlow

# 使い方
学習用画像は256x256であらかじめ揃える必要があります。train_dir, validation_dir内に画像種類別にディレクトリを作り、データセットの画像を置いておきます。二次元イラストとそれ以外の場合、

```
train_dir/nijigen/
train_dir/sanjigen/
validation_dir/nijigen
validation_dir/sanjigen

```
のような感じになります。サブディレクトリがそのままラベルになるので、試していませんが3種類以上もスクリプトの改変なしで動かせると思います。