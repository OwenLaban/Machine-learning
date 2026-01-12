# Music Year Prediction - End-to-End Regression Pipeline

## 📋 Deskripsi Proyek

Proyek ini merupakan implementasi end-to-end regression pipeline untuk memprediksi tahun rilis lagu berdasarkan fitur-fitur audio. Dataset yang digunakan berisi informasi audio dari berbagai lagu, dengan target variabel berupa tahun rilis (2001, 2002, dst.) dan fitur-fitur numerik yang merepresentasikan karakteristik audio seperti timbre dan properti suara lainnya.

## 🎯 Tujuan

Membangun model machine learning yang dapat memprediksi tahun rilis sebuah lagu dengan akurasi tinggi berdasarkan fitur-fitur audio yang tersedia.

## 📊 Dataset

- **Sumber**: Google Drive (midterm-regresi-dataset.csv)
- **Format**: CSV tanpa header
- **Struktur**:
  - Kolom pertama: Target variable (tahun rilis)
  - Kolom sisanya: Fitur-fitur audio (feature_1, feature_2, ..., feature_n)
- **Karakteristik**:
  - Semua fitur bertipe numerik
  - Fitur merepresentasikan karakteristik audio yang telah diekstrak dari sinyal musik

## 🔧 Teknologi dan Library

### Core Libraries
- **pandas**: Manipulasi dan analisis data
- **numpy**: Operasi numerik dan array
- **scikit-learn**: Machine learning algorithms dan tools
- **lightgbm**: Fast gradient boosting framework

### Visualization
- **matplotlib**: Plotting dan visualisasi
- **seaborn**: Visualisasi statistik yang lebih advanced

### Utility
- **gdown**: Download file dari Google Drive
- **warnings**: Suppress warning messages

## 📐 Pipeline Architecture

### 1. **Data Loading & Exploration**
   - Download dataset menggunakan gdown
   - Load data dengan pandas
   - Exploratory Data Analysis (EDA):
     - Statistik deskriptif
     - Distribusi target variable
     - Missing value check
     - Visualisasi data

### 2. **Data Preprocessing**

#### a. Outlier Detection & Handling
   - Metode: **IQR (Interquartile Range)**
   - Formula:
     ```
     Q1 = Kuartil pertama (25%)
     Q3 = Kuartil ketiga (75%)
     IQR = Q3 - Q1
     Lower Bound = Q1 - 1.5 × IQR
     Upper Bound = Q3 + 1.5 × IQR
     ```
   - Handling: **Capping/Winsorization**
     - Nilai di bawah lower bound → diganti dengan lower bound
     - Nilai di atas upper bound → diganti dengan upper bound

#### b. Feature Selection
   - Metode: **SelectKBest dengan f_regression**
   - Memilih top 50 fitur (atau semua jika kurang dari 50)
   - Scoring: F-statistic dari regresi univariat
   - Tujuan: Mengurangi dimensionalitas dan meningkatkan performa model

### 3. **Data Splitting**
   - **Train-Test Split**: 80-20
   - Strategi: Random split dengan random_state=42 untuk reproducibility
   - Train set: Untuk training model
   - Test set: Untuk evaluasi final

### 4. **Feature Scaling**
   - Metode: **RobustScaler**
   - Alasan pemilihan:
     - Lebih robust terhadap outliers dibanding StandardScaler
     - Menggunakan median dan IQR untuk scaling
     - Formula: `(X - median) / IQR`

### 5. **Model Training & Evaluation**

#### Models Evaluated
1. **Ridge Regression**
   - Linear regression dengan L2 regularization
   - Baik untuk mengatasi multicollinearity
   - Fast dan efficient untuk baseline
   
2. **Lasso Regression**
   - Linear regression dengan L1 regularization
   - Dapat melakukan feature selection otomatis
   - Good untuk sparse features
   
3. **LightGBM (Light Gradient Boosting Machine)**
   - State-of-the-art gradient boosting framework
   - **Extremely fast** training speed (10-20x faster than traditional GBM)
   - **High performance** dengan memory efficiency
   - Menggunakan leaf-wise tree growth
   - Built-in handling untuk categorical features
   - Optimal untuk dataset besar
   - **Dipilih karena**: Kecepatan training sangat cepat, cocok untuk Google Colab free tier

#### Evaluation Metrics
- **RMSE (Root Mean Squared Error)**
  - Formula: `√(Σ(y_pred - y_actual)² / n)`
  - Interpretasi: Average error dalam satuan tahun
  - Lebih sensitif terhadap outliers
  
- **MAE (Mean Absolute Error)**
  - Formula: `Σ|y_pred - y_actual| / n`
  - Interpretasi: Average absolute error
  - Lebih robust terhadap outliers
  
- **R² Score (Coefficient of Determination)**
  - Formula: `1 - (SS_res / SS_tot)`
  - Range: 0 hingga 1 (bisa negatif jika sangat buruk)
  - Interpretasi: Proporsi variansi yang dijelaskan model
    - R² > 0.8: Excellent
    - R² > 0.6: Good
    - R² > 0.4: Moderate
    - R² < 0.4: Poor

- **Cross-Validation R²**
  - 5-Fold Cross Validation
  - Memberikan estimasi performa yang lebih robust
  - Mean dan standard deviation dilaporkan

### 6. **Hyperparameter Tuning**
   - Metode: **GridSearchCV**
   - Cross-Validation: 5-fold
   - Scoring: R² score
   - Parameter grids untuk setiap model:

#### LightGBM
```python
{
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'num_leaves': [31, 50],
    'min_child_samples': [20, 30]
}
```

#### Random Forest
```python
{
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
```

#### Gradient Boosting
```python
{
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5],
    'min_samples_split': [2, 5]
}
```

#### Ridge/Lasso/ElasticNet
```python
{
    'alpha': [0.01, 0.1, 1.0, 10.0]
}
```

## 📈 Visualisasi

### 1. **Data Exploration Visualizations**
   - Distribution of Song Release Years
   - Boxplot of Release Years
   - Distribution of Sample Features
   - Top Features by Correlation

### 2. **Model Performance Visualizations**
   - Model Comparison - Test R² Score
   - Model Comparison - Test RMSE
   - Actual vs Predicted Scatter Plot
   - Residual Plot

## 🚀 Cara Menggunakan

### 1. Setup di Google Colab

```bash
# Buka Google Colab: https://colab.research.google.com/
# Upload file .ipynb atau copy-paste kode
```

### 2. Run All Cells

```python
# Klik Runtime > Run all
# atau tekan Ctrl+F9 (Windows) / Cmd+F9 (Mac)
```

### 3. Monitoring Progress

Notebook akan menampilkan progress untuk setiap section:
- ✓ Libraries imported
- ✓ Dataset downloaded
- ✓ Data cleaned
- ✓ Models trained
- ✓ Results visualized

## 📊 Expected Results

### Model Performance (Estimasi)

| Model | Test RMSE | Test MAE | Test R² | Training Speed |
|-------|-----------|----------|---------|----------------|
| Ridge Regression | ~8-10 years | ~6-8 years | 0.65-0.75 | ⚡ Very Fast |
| Lasso Regression | ~8-10 years | ~6-8 years | 0.65-0.75 | ⚡ Very Fast |
| LightGBM | ~5-7 years | ~4-5 years | 0.80-0.90 | ⚡⚡⚡ Ultra Fast |

**Why LightGBM?**
- ✅ **10-20x faster** than traditional Gradient Boosting
- ✅ **Memory efficient** - cocok untuk Colab free tier
- ✅ **High accuracy** - often best performer
- ✅ **GPU support** (optional) untuk speed boost
- ✅ **Native categorical handling**

*Note: Actual results may vary depending on the dataset*

## 🔍 Interpretasi Hasil

### R² Score Interpretation
- **R² > 0.8**: Model sangat baik, menjelaskan >80% variansi
- **R² > 0.6**: Model baik, menjelaskan >60% variansi
- **R² > 0.4**: Model moderat, ada room for improvement
- **R² < 0.4**: Model kurang baik, perlu feature engineering

### Error Metrics
- **RMSE/MAE rendah**: Model akurat dalam prediksi
- Contoh: MAE = 5 years → rata-rata prediksi meleset 5 tahun
- Untuk task prediksi tahun musik, error 5-8 tahun cukup reasonable

## 🎓 Penjelasan Teknis Mendalam

### Mengapa RobustScaler?
- **Robust terhadap outliers**: Menggunakan median dan IQR
- **StandardScaler** sensitif terhadap outliers karena menggunakan mean dan std
- **MinMaxScaler** juga sensitif karena menggunakan min dan max

### Mengapa SelectKBest?
- **Curse of dimensionality**: Terlalu banyak fitur dapat menurunkan performa
- **Computational efficiency**: Lebih cepat training dengan fitur lebih sedikit
- **Reduced overfitting**: Fitur irrelevant dapat menyebabkan overfitting

### Mengapa Cross-Validation?
- **Single split** bisa memberikan estimasi performa yang bias
- **K-fold CV** memberikan estimasi yang lebih reliable
- Menggunakan data lebih efisien (semua data digunakan untuk training dan validation)

### Model Selection Strategy
1. Mulai dengan **linear models** (Ridge, Lasso) untuk baseline cepat
2. Gunakan **LightGBM** untuk performa terbaik dengan kecepatan optimal
3. **Tune hyperparameters** pada model terbaik
4. **Compare** menggunakan multiple metrics

### Mengapa LightGBM Dipilih?

#### Keunggulan LightGBM:
1. **Speed**: 10-20x lebih cepat dari Gradient Boosting tradisional
2. **Memory Efficient**: Menggunakan histogram-based algorithm
3. **Accuracy**: Sering memberikan hasil terbaik
4. **Scalability**: Excellent untuk dataset besar
5. **Features**:
   - Leaf-wise tree growth (lebih optimal)
   - Built-in support untuk missing values
   - Native categorical feature support
   - GPU acceleration support (opsional)

#### Perbandingan Training Time:
- **Gradient Boosting (sklearn)**: ~5-10 menit
- **Random Forest**: ~3-5 menit  
- **LightGBM**: ~30 detik - 1 menit ⚡
- **Ridge/Lasso**: ~10-20 detik

#### Technical Details:
- **Histogram-based algorithm**: Membagi continuous features ke bins
- **Leaf-wise growth**: Lebih dalam tapi controlled dengan max_depth
- **Gradient-based One-Side Sampling (GOSS)**: Skip data points dengan gradients kecil
- **Exclusive Feature Bundling (EFB)**: Bundle sparse features

## 🐛 Troubleshooting

### Issue: Download Error
```python
# Solusi: Pastikan file ID benar dan file accessible
# Atau download manual lalu upload ke Colab
```

### Issue: Memory Error
```python
# Solusi: Kurangi jumlah features yang diselect
# Atau gunakan model yang lebih lightweight
```

### Issue: Long Training Time
```python
# Solusi:
# 1. Reduce n_estimators in Random Forest/Gradient Boosting
# 2. Reduce parameter grid size in GridSearchCV
# 3. Use n_jobs=-1 untuk parallel processing
```

## 📚 Referensi dan Sumber Belajar

### Scikit-learn Documentation
- [Regression](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
- [Model Selection](https://scikit-learn.org/stable/model_selection.html)
- [Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)

### LightGBM Documentation
- [LightGBM Official Docs](https://lightgbm.readthedocs.io/)
- [Parameters Tuning](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html)
- [Python API](https://lightgbm.readthedocs.io/en/latest/Python-API.html)

### Konsep Machine Learning
- [Understanding R² Score](https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score)
- [Cross-Validation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)

### Best Practices
- [Hyperparameter Tuning](https://scikit-learn.org/stable/modules/grid_search.html)
- [Model Evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html)

## 💡 Insights dan Pembelajaran

### Key Takeaways
1. **Preprocessing is crucial**: Outlier handling dan feature selection sangat mempengaruhi hasil
2. **LightGBM is the winner**: Fast, accurate, dan memory-efficient
3. **Speed matters**: LightGBM 10-20x lebih cepat dari traditional boosting
4. **Hyperparameter tuning matters**: Bisa meningkatkan performa 5-10%
5. **Multiple metrics**: Jangan hanya fokus pada satu metric

### Possible Improvements
1. **LightGBM GPU**: Enable GPU acceleration untuk speed 5-10x lebih cepat
2. **Feature Engineering**: Buat fitur baru dari kombinasi fitur existing
3. **Deep Learning**: Coba neural networks untuk complex patterns
4. **Stacking/Blending**: Kombinasi LightGBM dengan models lain
5. **Advanced LightGBM tuning**: Dart mode, categorical features optimization
6. **Time-based validation**: Jika data punya time component

## 📝 Kesimpulan

Pipeline ini mendemonstrasikan:
- ✅ Complete end-to-end workflow
- ✅ Proper data preprocessing
- ✅ Multiple model comparison
- ✅ Hyperparameter tuning
- ✅ Comprehensive evaluation
- ✅ Clear visualization dan interpretation

Model yang dihasilkan dapat memprediksi tahun rilis lagu dengan reasonable accuracy, menunjukkan bahwa fitur-fitur audio memang mengandung informasi temporal tentang kapan musik dibuat.

## 👨‍💻 Author

Dikembangkan untuk Midterm Assignment - Machine Learning Course

## 📄 License

Educational purposes only

---

**Happy Modeling! 🚀🎵**