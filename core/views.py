import re
from collections import Counter
from pathlib import Path

import nltk
import pandas as pd
import plotly.express as px
from django.conf import settings
from django.shortcuts import render
from nltk.sentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

DATASET_PATH = Path(settings.BASE_DIR) / 'books-15k.csv'


def ensure_vader_lexicon():
    try:
        nltk.data.find('sentiment/vader_lexicon')
        return True
    except LookupError:
        try:
            nltk.download('vader_lexicon', quiet=True, raise_on_error=True)
            return True
        except Exception:
            return False


def parse_helpfulness(value):
    if pd.isna(value):
        return 0.0
    text = str(value).strip()
    if not text or text == 'nan':
        return 0.0
    if '/' not in text:
        try:
            return float(text)
        except ValueError:
            return 0.0
    try:
        helpful, total = text.split('/', 1)
        helpful = float(helpful)
        total = float(total)
        if total == 0:
            return 0.0
        return helpful / total
    except ValueError:
        return 0.0


def generate_wordcloud(df):
    text_values = df['review/text'].fillna('').astype(str).str.lower()
    text = ' '.join(text_values)
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = [word for word in text.split() if len(word) > 2 and word not in {'book', 'books', 'read', 'reads', 'reading', 'story', 'people', 'really', 'would'}]
    if not words:
        words = ['book', 'reading', 'review']
    wc = WordCloud(width=900, height=400, background_color='white', colormap='viridis')
    wc.generate(' '.join(words))
    output_path = Path(settings.BASE_DIR) / 'core' / 'static' / 'core' / 'wordcloud.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wc.to_file(str(output_path))
    return output_path.name


def build_dashboard_context():
    df = pd.read_csv(DATASET_PATH)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df['review/score'] = pd.to_numeric(df['review/score'], errors='coerce')
    df['review/time'] = pd.to_numeric(df['review/time'], errors='coerce')
    df['review/date'] = pd.to_datetime(df['review/time'], unit='s', errors='coerce')
    df['review/helpfulness_ratio'] = df['review/helpfulness'].apply(parse_helpfulness)

    missing = df.isna().sum().sort_values(ascending=False)
    missing_table = missing.reset_index()
    missing_table.columns = ['Coluna', 'Quantidade']
    missing_table['Proporcao'] = (missing_table['Quantidade'] / len(df) * 100).round(2)

    price_mean = round(float(df['Price'].mean()), 2)
    books_zero_price = int((df['Price'] == 0).sum())
    score_mean = float(df['review/score'].mean())
    score_median = float(df['review/score'].median())
    score_std = float(df['review/score'].std())
    total_unique_books = df['Title'].nunique()
    mean_reviews_per_book = float(df['Title'].value_counts().mean())

    score_counts = df['review/score'].value_counts().sort_index().rename_axis('Nota').reset_index(name='Frequencia')
    score_counts['Nota'] = score_counts['Nota'].astype(int)

    top_books = df['Title'].value_counts().head(10).reset_index()
    top_books.columns = ['Titulo', 'Avaliacoes']

    top_users = df['User_id'].value_counts().head(10).reset_index()
    top_users.columns = ['Usuario', 'Avaliacoes']

    top_books_chart = px.bar(
        top_books,
        x='Titulo',
        y='Avaliacoes',
        title='Top 10 livros mais avaliados',
        labels={'Titulo': 'Livro', 'Avaliacoes': 'Avaliações'},
        color='Avaliacoes',
        color_continuous_scale='Viridis'
    )
    top_books_chart.update_layout(template='plotly_white', height=420, margin=dict(l=20, r=20, t=50, b=10), xaxis_tickangle=-45)

    top_users_chart = px.bar(
        top_users,
        x='Usuario',
        y='Avaliacoes',
        title='Top 10 usuários com mais avaliações',
        labels={'Usuario': 'Usuário', 'Avaliacoes': 'Avaliações'},
        color='Avaliacoes',
        color_continuous_scale='Viridis'
    )
    top_users_chart.update_layout(template='plotly_white', height=420, margin=dict(l=20, r=20, t=50, b=10), xaxis_tickangle=-45)

    price_distribution = df['Price'].dropna().copy()
    price_distribution = price_distribution[price_distribution <= 200].copy()

    price_box_chart = px.histogram(
        price_distribution,
        x=price_distribution,
        nbins=30,
        title='Distribuição de preços dos livros',
        color_discrete_sequence=['#4f5ee0']
    )
    price_box_chart.update_traces(marker=dict(line=dict(color='rgba(79,94,224,0.7)', width=1)))
    price_box_chart.update_xaxes(title='Preço do livro', showgrid=True, gridcolor='rgba(0,0,0,0.08)')
    price_box_chart.update_yaxes(title='Frequência', showgrid=True, gridcolor='rgba(0,0,0,0.08)')
    price_box_chart.update_layout(
        template='plotly_white',
        height=320,
        margin=dict(l=20, r=20, t=50, b=10),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )

    useful_reviews = df.sort_values('review/helpfulness_ratio', ascending=False).head(5)[
        ['Title', 'profileName', 'review/helpfulness', 'review/helpfulness_ratio', 'review/score', 'review/summary']
    ].copy()
    useful_reviews.columns = ['Titulo', 'Usuario', 'Utilidade', 'TaxaUtilidade', 'Nota', 'Resumo']
    useful_reviews['TaxaUtilidade'] = useful_reviews['TaxaUtilidade'].round(2)

    years = df['review/date'].dt.year.value_counts().sort_index()
    year_df = years.reset_index()
    year_df.columns = ['Ano', 'Avaliacoes']

    score_chart = px.bar(
        score_counts,
        x='Nota',
        y='Frequencia',
        title='Distribuição das notas (1 a 5)',
        labels={'Nota': 'Nota', 'Frequencia': 'Quantidade'},
        color='Nota',
        color_continuous_scale='Viridis'
    )
    score_chart.update_layout(template='plotly_white', height=420, margin=dict(l=20, r=20, t=50, b=10))

    price_score_chart = px.scatter(
        df,
        x='Price',
        y='review/score',
        opacity=0.5,
        title='Preço vs. nota atribuída',
        labels={'Price': 'Preço do livro', 'review/score': 'Nota'},
        color='review/score',
        color_continuous_scale='Viridis'
    )
    price_score_chart.update_layout(template='plotly_white', height=420, margin=dict(l=20, r=20, t=50, b=10))

    time_chart = px.line(
        year_df,
        x='Ano',
        y='Avaliacoes',
        title='Evolução temporal das avaliações por ano',
        markers=True,
        line_shape='spline'
    )
    time_chart.update_layout(template='plotly_white', height=420, margin=dict(l=20, r=20, t=50, b=10))

    sentiment = SentimentIntensityAnalyzer() if ensure_vader_lexicon() else None
    sentiment_labels = []
    review_texts = df['review/text'].fillna('').astype(str)
    for text in review_texts:
        if sentiment is None:
            sentiment_labels.append('Neutro')
            continue
        score = sentiment.polarity_scores(text)
        if score['compound'] > 0.05:
            sentiment_labels.append('Positivo')
        elif score['compound'] < -0.05:
            sentiment_labels.append('Negativo')
        else:
            sentiment_labels.append('Neutro')
    sentiment_counts = Counter(sentiment_labels)
    sentiment_counts = {
        'Positivo': sentiment_counts.get('Positivo', 0),
        'Neutro': sentiment_counts.get('Neutro', 0),
        'Negativo': sentiment_counts.get('Negativo', 0),
    }

    sentiment_df = pd.DataFrame([
        {'Sentimento': 'Positivo', 'Quantidade': sentiment_counts['Positivo']},
        {'Sentimento': 'Neutro', 'Quantidade': sentiment_counts['Neutro']},
        {'Sentimento': 'Negativo', 'Quantidade': sentiment_counts['Negativo']},
    ])
    sentiment_chart = px.bar(
        sentiment_df,
        x='Sentimento',
        y='Quantidade',
        title='Distribuição do sentimento das avaliações',
        labels={'Sentimento': 'Sentimento', 'Quantidade': 'Quantidade'},
        color='Sentimento',
        color_discrete_map={'Positivo': '#39ff14', 'Neutro': '#b0b7c3', 'Negativo': '#ff4d4d'}
    )
    sentiment_chart.update_layout(template='plotly_white', height=340, margin=dict(l=20, r=20, t=50, b=10))

    generate_wordcloud(df)

    dtypes_table = df.dtypes.astype(str).reset_index().rename(columns={'index': 'Coluna', 0: 'Tipo'})
    dtypes_counts = dtypes_table['Tipo'].value_counts().reset_index()
    dtypes_counts.columns = ['Tipo', 'Quantidade']

    dtypes_chart = px.bar(
        dtypes_counts,
        x='Tipo',
        y='Quantidade',
        color='Tipo',
        color_discrete_map={
            'str': '#d9b95c',
            'float64': '#e74c3c',
            'int64': '#f39c12',
            'datetime64[us]': '#f4c7b5',
        },
        title=''
    )
    dtypes_chart.update_traces(textposition='none', width=0.7)
    dtypes_chart.update_layout(
        template='plotly_white',
        height=260,
        margin=dict(l=50, r=20, t=10, b=10),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        xaxis_title='',
        yaxis_title='',
        font=dict(size=12),
        bargap=0.35
    )
    dtypes_chart.update_xaxes(showgrid=True, gridcolor='rgba(0,0,0,0.12)', title_font=dict(size=12), tickfont=dict(size=11))
    dtypes_chart.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.12)', title_font=dict(size=12), tickfont=dict(size=11))

    context = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'dtypes': dtypes_table.to_dict(orient='records'),
        'dtypes_chart': dtypes_chart.to_html(full_html=False, include_plotlyjs='cdn'),
        'missing_table': missing_table.to_dict(orient='records'),
        'price_mean': price_mean,
        'books_zero_price': books_zero_price,
        'score_mean': round(score_mean, 2),
        'score_median': round(score_median, 2),
        'score_std': round(score_std, 2),
        'total_unique_books': total_unique_books,
        'mean_reviews_per_book': round(mean_reviews_per_book, 2),
        'top_books': top_books.to_dict(orient='records'),
        'top_books_chart': top_books_chart.to_html(full_html=False, include_plotlyjs='cdn'),
        'top_users': top_users.to_dict(orient='records'),
        'top_users_chart': top_users_chart.to_html(full_html=False, include_plotlyjs='cdn'),
        'useful_reviews': useful_reviews.to_dict(orient='records'),
        'score_counts': score_counts.to_dict(orient='records'),
        'year_data': year_df.to_dict(orient='records'),
        'score_chart': score_chart.to_html(full_html=False, include_plotlyjs='cdn'),
        'price_score_chart': price_score_chart.to_html(full_html=False, include_plotlyjs='cdn'),
        'price_box_chart': price_box_chart.to_html(full_html=False, include_plotlyjs='cdn'),
        'time_chart': time_chart.to_html(full_html=False, include_plotlyjs='cdn'),
        'sentiment_chart': sentiment_chart.to_html(full_html=False, include_plotlyjs='cdn'),
        'sentiment_counts': sentiment_counts,
        'positive_reviews_ratio': round((sentiment_counts['Positivo'] / sum(sentiment_counts.values())) * 100, 2),
        'negative_reviews_ratio': round((sentiment_counts['Negativo'] / sum(sentiment_counts.values())) * 100, 2),
    }
    return context


def compute_sentiment_summary(df, sample_size=None):
    analyzer = SentimentIntensityAnalyzer() if ensure_vader_lexicon() else None
    labels = []
    texts = df['review/text'].fillna('').astype(str)
    if sample_size is not None:
        texts = texts.head(sample_size)

    for text in texts:
        if analyzer is None:
            labels.append('Neutro')
            continue
        score = analyzer.polarity_scores(text)
        if score['compound'] > 0.05:
            labels.append('Positivo')
        elif score['compound'] < -0.05:
            labels.append('Negativo')
        else:
            labels.append('Neutro')

    count = Counter(labels)
    total = sum(count.values())
    return {
        'Positivo': count.get('Positivo', 0),
        'Neutro': count.get('Neutro', 0),
        'Negativo': count.get('Negativo', 0),
        'total': total,
        'percentual_positivo': round((count.get('Positivo', 0) / total) * 100, 2) if total else 0,
        'percentual_negativo': round((count.get('Negativo', 0) / total) * 100, 2) if total else 0,
    }


def home(request):
    return render(request, 'core/index.html', build_dashboard_context())


def sentimento(request):
    df = pd.read_csv(DATASET_PATH)
    context = compute_sentiment_summary(df)
    context['dataset_total'] = len(df)
    context['title'] = 'Análise de Sentimento'
    return render(request, 'core/sentimento.html', context)
