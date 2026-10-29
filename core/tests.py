import pandas as pd
from django.test import SimpleTestCase

from core.views import compute_sentiment_summary


class HomePageTests(SimpleTestCase):
    def test_homepage_loads(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Amazon Book Rating Dataset')
        self.assertIn('total_rows', response.context)
        self.assertGreater(response.context['total_rows'], 0)
        self.assertTrue(response.context['dtypes'])
        self.assertIn('Coluna', response.context['dtypes'][0])
        self.assertIn('Tipo', response.context['dtypes'][0])

    def test_sentiment_page_loads(self):
        response = self.client.get('/sentimento/')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Análise de Sentimento')
        self.assertIn('Positivo', response.context)
        self.assertIn('Negativo', response.context)
        self.assertIn('Neutro', response.context)

    def test_sentiment_summary_uses_full_dataset(self):
        df = pd.DataFrame({'review/text': ['This book is excellent!'] * 15000})
        summary = compute_sentiment_summary(df, sample_size=None)
        self.assertEqual(summary['total'], 15000)
        self.assertEqual(summary['Positivo'], 15000)
