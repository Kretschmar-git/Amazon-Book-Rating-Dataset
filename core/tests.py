from django.test import SimpleTestCase


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
