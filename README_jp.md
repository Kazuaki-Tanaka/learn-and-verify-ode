# Learn and Verify ODE

**ニューラルネットワークを用いた常微分方程式の厳密な解の包含のためのMATLABパッケージ**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MATLAB](https://img.shields.io/badge/MATLAB-R2025b-blue.svg)](https://www.mathworks.com/products/matlab.html)

本リポジトリは以下の論文のコードを提供します：

> **"Learn and Verify: A Rigorous Approach to Enclosing Solutions of Differential Equations through Deep Learning with Sub- and Super-solutions"**  
> Kazuaki Tanaka and Kohei Yatabe

## 📖 概要

従来のPhysics-Informed Neural Networks (PINNs) は微分方程式の解を近似できますが、厳密な誤差範囲を保証することはできません。我々の **Learn and Verify**（学習と検証）アプローチは、以下の手順によってこの課題を解決します：

1.  **学習 (Learn)**: ニューラルネットワークを用いて優解 (super-solution) と劣解 (sub-solution) を学習します
2.  **検証 (Verify)**: 区間演算 (INTLAB) を用いて学習した候補を厳密に検証し、真の解が優解と劣解の間に存在することを数学的に保証します

### 主な特徴

- ✅ **数学的に厳密**: 誤差範囲が保証された解の包含 (enclosure) を提供
- 🧠 **ニューラルネットワークベース**: 優解および劣解の学習に深層学習を活用
- 🔬 **区間演算**: INTLABを用いた信頼性の高い検証
- ⚡ **柔軟なアーキテクチャ**: 様々な常微分方程式 (ODE) に適用可能（現状は1次元スカラーODEのみ対応）

## 🚀 クイックスタート

### 前提条件

- **MATLAB** R2025b
- **INTLAB** 14 ([ダウンロードはこちら](http://www.ti3.tu-harburg.de/rump/intlab/))
- **Deep Learning Toolbox** (ニューラルネットワーク機能のため)

> **注意**: 上記以外のバージョンでの動作は保証しません。ただし、Deep Learning ToolboxとINTLABが動作するMATLABバージョンであれば、おそらく動作します。

### インストール

1. 本リポジトリをクローンします：
```bash
git clone https://github.com/Kazuaki-Tanaka/learn-and-verify-ode.git
cd learn-and-verify-ode
```

2. MATLABでセットアップスクリプトを実行し、パスに追加します：
```matlab
cd learn-and-verify-ode
setup
```

3. INTLABがインストールされているか確認します：
```matlab
intvalinit('DisplayInfsup')
```

### 基本的な使い方

使用例は [`examples/`](examples/) フォルダを参照してください：

- [`simple_logistic.m`](examples/simple_logistic.m) - 古典的なロジスティック方程式の例
- [`general_logistic.m`](examples/general_logistic.m) - 時変係数を持つロジスティック方程式の例

```matlab
% 例の実行
cd examples
run simple_logistic.m
```

## 📂 リポジトリ構成

```
learn-and-verify-ode/
├── +OneDim/                    # メインパッケージ (MATLAB package)
│   ├── NN.m                    # ニューラルネットワークモデル
│   ├── RigorousNN.m            # INTLABを用いた厳密検証
│   ├── IVPsolver.m             # 近似解ソルバー
│   ├── IVPsolver2.m            # 優解・劣解ソルバー
│   └── Optimizer.m             # ADAM オプティマイザ
│
└── examples/                   # 使用例
    ├── simple_logistic.m       # 古典的なロジスティック方程式
    └── general_logistic.m      # 時変係数を持つロジスティック方程式
```

## 🧩 主要コンポーネント

### 1. ニューラルネットワークモデル (`NN.m`)

- カスタマイズ可能なアーキテクチャを持つ全結合ネットワーク
- 正弦波活性化関数のためのSIREN初期化
- 最大4階までの自動微分をサポート
- 初期条件のハード制約 (Hard constraint) をサポート

### 2. 厳密な検証 (`RigorousNN.m`)

- 区間演算に基づいた検証
- 厳密な包含のための適応的な細分化 (Adaptive subdivision)
- 詳細な診断情報の提供

### 3. ソルバー

- **`IVPsolver.m`**: ペナルティ正則化を用いて近似解を学習
- **`IVPsolver2.m`**: 優解と劣解を同時に学習

### 4. オプティマイザ (`Optimizer.m`)

- ADAMを使用した学習
- サンプリングモード: random, grid, region, auto（autoはrandomからgridに自動切り替え）

## 📝 ライセンス

本プロジェクトは MIT ライセンスの下で公開されています。詳細は [LICENSE](LICENSE) ファイルをご覧ください。

## 🙏 謝辞

本研究は、JST創発的研究支援事業 (FOREST) (Grant Number JPMJFR202S) の支援を受けています。

## 📧 連絡先

ご質問や問題がある場合は以下までご連絡ください：
- **Kazuaki Tanaka**: tanaka@ims.sci.waseda.ac.jp
- **GitHub Issues**: [https://github.com/Kazuaki-Tanaka/learn-and-verify-ode/issues](https://github.com/Kazuaki-Tanaka/learn-and-verify-ode/issues)

---

**キーワード**: 微分方程式, ニューラルネットワーク, 深層学習, 厳密な包含, 区間演算, 優解・劣解, 精度保証付き数値計算
