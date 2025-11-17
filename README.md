# X Community Notes Data Science Analysis

A comprehensive data science project analyzing X (Twitter) Community Notes using advanced machine learning and data analysis techniques.

## 📋 Project Overview

This project performs in-depth analysis of X Community Notes, a crowdsourced fact-checking system. The analysis includes:

1. **Topic Classification**: 15-topic classification system covering conflicts, geopolitics, and general categories
2. **Community Detection**: Network analysis to identify communities and their topic specializations
3. **Exploratory Data Analysis**: Comprehensive statistical analysis and visualizations

### Key Features

- **15-Topic Classification System**: Covers Ukraine Conflict, Gaza Conflict, Syria War, Iran, China-Taiwan, China Influence, Other Conflicts, Scams, Health/Medical, Climate/Environment, Politics, Technology, Economics, Entertainment, and Immigration
- **TF-IDF + Logistic Regression**: Efficient and accurate classification model
- **Community Analysis**: Louvain algorithm for community detection with topic specialization analysis
- **Comprehensive Visualizations**: Professional charts and dashboards for insights

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Required Python packages (see `requirements.txt` or install instructions below)

### Installation

1. Clone or download this repository
2. Install required dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn networkx langdetect
```

Optional (for better language detection):
```bash
pip install textblob
```

### Data Setup

Place your Community Notes data in the following structure:
```
data/
  notes/
    notes-00000.tsv
```

The TSV file should contain columns including:
- `noteId`: Unique identifier for each note
- `summary`: Text content of the note
- `createdAtMillis`: Timestamp (optional, for temporal analysis)

## 📁 Project Structure

```
.
├── code/
│   ├── classification/
│   │   ├── topic_classifier.py          # Main classification module
│   │   └── analyze_classification_results.py  # Results analysis and visualization
│   ├── clustering/
│   │   ├── communities_analysis.py     # Community detection and analysis
│   │   └── louvain_communities.ipynb    # Jupyter notebook for community analysis
│   ├── eda/
│   │   └── eda_analysis.py              # Exploratory data analysis
│   ├── demo_workflow.py                 # Demo script showcasing workflow
│   └── generate_report.py               # Report generator for presentations
├── run_analysis.py                      # Main entry point script
├── plots/
│   ├── topic_classification/          # Classification visualizations
│   ├── comuunity_detection/            # Community analysis visualizations
│   └── eda/                            # EDA visualizations
└── README.md                           # This file
```

## 🎯 Usage

### Quick Start (Recommended)

The easiest way to run the analysis is using the main entry point script:

```bash
# Run complete pipeline (classification + analysis + report)
python run_analysis.py all --max-notes 10000

# Or run individual components
python run_analysis.py classify --max-notes 10000
python run_analysis.py analyze
python run_analysis.py report

# Run demo workflow
python run_analysis.py demo --quick
```

### Demo Workflow

For a quick demonstration of the complete workflow:

```bash
# Quick demo with 10K notes
python code/demo_workflow.py --quick

# Full demo with all notes
python code/demo_workflow.py --full

# Analysis only (on existing results)
python code/demo_workflow.py --analysis-only
```

### Programmatic Usage

#### Topic Classification

```python
from code.classification.topic_classifier import CustomTopicClassifier

# Initialize classifier
classifier = CustomTopicClassifier(data_path="data", output_dir="results")

# Run complete pipeline
results = classifier.run_complete_pipeline(
    max_notes=10000,      # Limit to 10K notes for testing
    english_only=True,    # Filter to English-only notes
    force_refilter=False  # Use cached English-filtered data if available
)
```

#### Classification Results Analysis

```python
from code.classification.analyze_classification_results import ClassificationAnalyzer

# Initialize analyzer
analyzer = ClassificationAnalyzer(results_dir="custom_topic_results")

# Load latest results and run complete analysis
results = analyzer.run_complete_analysis()
```

#### Generate Summary Report

```python
from code.generate_report import ReportGenerator

# Generate presentation-ready report
generator = ReportGenerator(output_dir="reports")
report_file = generator.generate_report(results_dir="custom_topic_results")
```

#### Community Analysis

```python
from code.clustering.communities_analysis import SpecializedVisualizationsDevide

# Initialize visualizer
visualizer = SpecializedVisualizationsDevide()

# Run all visualizations
results = visualizer.run_all_visualizations()
```

## 📊 Output Files

### Classification Outputs

- `classified_notes_*.csv`: Full classification results with topic labels and confidence scores
- `trained_topic_model_*.pkl`: Saved trained model for reuse
- `seed_terms_*.json`: Seed terms used for each topic
- `classification_summary_*.json`: Complete metadata and statistics

### Analysis Outputs

- `classification_analytics/`: Classification analysis charts and visualizations
- `reports/`: Generated summary reports (text and markdown formats)
- `plots/topic_classification/`: Topic distribution, confidence analysis, temporal trends
- `plots/comuunity_detection/`: Community structure, topic leadership matrices

### Generated Reports

The report generator creates presentation-ready summaries:
- `analysis_report_*.txt`: Text format report
- `analysis_report_*.md`: Markdown format report

Reports include:
- Executive summary with key metrics
- Topic distribution analysis
- Confidence statistics
- Methodology overview
- Key insights and findings

## 🔬 Methodology

### Topic Classification

1. **Seed Term Matching**: Initial labels assigned based on keyword matching across 15 topic categories
2. **Model Training**: TF-IDF vectorization (unigrams + bigrams) + Logistic Regression with class balancing
3. **Classification**: Apply trained model to all notes with confidence scores
4. **Analysis**: Comprehensive statistics and visualizations

### Community Detection

- Louvain algorithm for community detection
- Topic specialization analysis per community
- Diversity and engagement metrics
- Topic leadership matrices

## 📈 Key Results

- **Coverage Rate**: 85.6% (vs ~40-60% typical for zero-shot methods)
- **Model Accuracy**: 81.4% (excellent for 16-class problem)
- **Processing Speed**: ~1000 notes/second
- **Topics Covered**: 15 comprehensive categories

## 🛠️ Technical Details

### Classification Model

- **Vectorization**: TF-IDF with 50,000 max features
- **N-grams**: Unigrams and bigrams
- **Classifier**: Logistic Regression (LBFGS solver, OVR multi-class)
- **Class Balancing**: Automatic class weight balancing

### Performance

- Efficient memory usage (~337MB vs 5GB+ for neural models)
- Fast training and inference
- Scalable to millions of notes

## 📝 Notes

- The classifier supports English-only filtering for better accuracy
- Cached English-filtered datasets are automatically saved for faster subsequent runs
- All outputs include timestamps for version tracking

## 🤝 Contributing

This is a university data science project. For questions or improvements, please contact the project team.

## 📄 License

This project is for academic/research purposes.

## 👥 Authors

Data Science Project Team - Year 4

---

**Last Updated**: 2024

