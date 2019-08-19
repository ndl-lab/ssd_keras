このリポジトリには、見開きのページをのど元で分割するための学習プログラムと推論プログラムが含まれています。
このプログラムは以下のリポジトリを改変して作成しています。

[rykov8's repo](https://github.com/rykov8/ssd_keras)

# inference
inference_inputディレクトリにのど元を分割したい画像を入れ、inference.pyを実行する。

inference_outputディレクトリに分割後の画像が出力される。


# Training

1.学習ファイルの準備
学習させたい画像ファイルをtraining/imgに、
のど位置情報をtraining/image.tsvにそれぞれ用意しておく。
※例では
(ファイル名)\t(中心からのずれの割合)
としましたが、tsvの形式に応じてtraining/make_pkl_for_page.pyをカスタマイズしてください。

training/size_convertion.py


で画像のサイズを300*300に変換しておく

2.pklの生成
make_pkl_for_page.pyを実行し、page_layout.pklを生成しておく。


3.学習
train.pyを実行し、学習を開始する。
checkpointsディレクトリに学習済weightsファイルが生成される。