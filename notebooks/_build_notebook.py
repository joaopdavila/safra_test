"""Build the canonical answers notebook from a single Python source.

Run with: ``uv run python notebooks/_build_notebook.py``

This keeps the .ipynb generation deterministic and reviewable in diffs. The
notebook can also be edited directly in Jupyter — this script is only the
initial generator.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB_PATH = Path(__file__).parent / "01_prova_safra.ipynb"


def md(text: str) -> dict:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> dict:
    return nbf.v4.new_code_cell(text)


CELLS = [
    md(
        "# Prova de proficiência Python — Safra Asset\n"
        "## v2 (2026) — refeita sobre o pacote `safra_quant`\n\n"
        "**Autor:** João Pedro Tavares d'Ávila · **Original:** 04/10/2022 "
        "(`archive/jptd_2022.ipynb`)\n\n"
        "Esta versão move toda a lógica para `src/safra_quant/`, cobre o pacote com "
        "`pytest`/`mypy --strict`/`ruff`, e corrige os bugs financeiros identificados "
        "na revisão da v1. O notebook fica intencionalmente fino — cada questão é "
        "uma chamada à API testada."
    ),
    code(
        "import os\n"
        "import pandas as pd\n"
        "\n"
        "from safra_quant import indicators, plots, returns, stats, strategy\n"
        "from safra_quant._demo import get_prices_with_fallback, is_demo_mode\n"
        "from safra_quant.benchmark import rebase_to\n"
        "from safra_quant.universe import IBOVESPA_TICKER, IBRX50_SEP_2022\n"
        "\n"
        "pd.set_option('display.max_columns', 12)\n"
        "pd.set_option('display.width', 140)\n"
        "\n"
        "if is_demo_mode():\n"
        "    print('⚠️  SAFRA_QUANT_DEMO=1 — usando dados sintéticos determinísticos.')\n"
        "    print('   Rode sem a variável e com acesso à Yahoo Finance para dados reais.')\n"
    ),
    md(
        "## Q1 — Preços de fechamento IBRX 50 (2017-11-01 → 2022-01-01)\n\n"
        "Janela ajustada ao enunciado (a v1 começava em 2017-01-01). NaNs no meio da "
        "série são preenchidos por interpolação linear (que é equivalente à média dos "
        "vizinhos quando há um único ponto faltante); `limit_direction='both'` cobre "
        "também as caudas."
    ),
    code(
        "prices, used_demo = get_prices_with_fallback(\n"
        "    list(IBRX50_SEP_2022), start='2017-11-01', end='2022-01-01',\n"
        ")\n"
        "print(f'Modo demo: {used_demo}')\n"
        "print(f'Shape: {prices.shape}  |  NaN restantes: {prices.isna().values.any()}')\n"
        "prices.tail()\n"
    ),
    md(
        "## Q2 — Últimos 12 dias + retornos log × percentuais\n\n"
        "**Correção sobre a v1**: os retornos são calculados na série em ordem "
        "cronológica e só então invertidos para exibição. A v1 invertia primeiro e "
        "chamava `.diff()`, o que trocava o sinal de todos os retornos."
    ),
    code(
        "log_r = returns.log_returns(prices)\n"
        "pct_r = returns.pct_returns(prices)\n"
        "\n"
        "last_12 = prices.iloc[-12:].iloc[::-1]\n"
        "last_12.iloc[:, :6]\n"
    ),
    code(
        "log_r.iloc[-12:].iloc[::-1].iloc[:, :6]\n"
    ),
    code(
        "pct_r.iloc[-12:].iloc[::-1].iloc[:, :6]\n"
    ),
    code(
        "diff = (log_r - pct_r).iloc[-12:].iloc[::-1].iloc[:, :6]\n"
        "print('log(1+r) - r — a diferença é O(r²), portanto desprezível para retornos diários:')\n"
        "diff\n"
    ),
    md(
        "## Q3 — BPAC11 e ITUB4: preço, variação e distribuição\n\n"
        "Cada ação produz uma figura com três painéis (preço, retornos diários, "
        "histograma de retornos). Bins do histograma escolhidos por Freedman-Diaconis."
    ),
    code(
        "for ticker in ('BPAC11.SA', 'ITUB4.SA'):\n"
        "    p = prices[ticker]\n"
        "    r = returns.log_returns(p)\n"
        "    fig = plots.price_variation_distribution(p, r, title_prefix=ticker)\n"
        "    fig.show()\n"
    ),
    md(
        "## Q4 — Retornos mensais e trimestrais (+ IBOVESPA)\n\n"
        "**Correção sobre a v1**: agregamos os preços com `resample('ME'/'QE')` e "
        "calculamos retornos sobre os preços agregados. A v1 baixava `interval='1mo'` "
        "diretamente e depois interpolava NaN entre os retornos, distorcendo média/var."
    ),
    code(
        "ibov, _ = get_prices_with_fallback([IBOVESPA_TICKER], '2017-11-01', '2022-01-01')\n"
        "ibov_series = ibov[IBOVESPA_TICKER]\n"
        "\n"
        "panel = pd.DataFrame({\n"
        "    'BPAC11': prices['BPAC11.SA'],\n"
        "    'ITUB4':  prices['ITUB4.SA'],\n"
        "    'IBOV':   ibov_series,\n"
        "})\n"
        "\n"
        "monthly = returns.resample_returns(panel, 'ME', kind='log').dropna()\n"
        "quarterly = returns.resample_returns(panel, 'QE', kind='log').dropna()\n"
        "\n"
        "print('Retornos mensais (head):')\n"
        "monthly.head()\n"
    ),
    code(
        "def summarize(df):\n"
        "    return df.agg(['mean', 'median', 'var',\n"
        "                   lambda s: s.quantile(0.25),\n"
        "                   lambda s: s.quantile(0.75)]).rename(\n"
        "        index={'<lambda_0>': 'q1', '<lambda_1>': 'q3'})\n"
        "\n"
        "print('=== Estatísticas mensais ===')\n"
        "print(summarize(monthly).round(4))\n"
        "print()\n"
        "print('=== Estatísticas trimestrais ===')\n"
        "print(summarize(quarterly).round(4))\n"
    ),
    md(
        "## Q5 — Estratégia RSI(21), entradas em 25/75, 2018-01-01 → 2021-12-31\n\n"
        "**Correções sobre a v1**: (i) RSI sobre `Adj Close` (não `Close`); (ii) "
        "PnL via curva de equity com posição marcada a mercado e fechamento explícito "
        "no último dia; (iii) implementação do RSI vetorizada e testada."
    ),
    code(
        "WINDOW = (slice('2018-01-01', '2021-12-31'))\n"
        "\n"
        "results = {}\n"
        "for ticker in ('BPAC11.SA', 'ITUB4.SA'):\n"
        "    px = prices[ticker].loc[WINDOW]\n"
        "    strat = strategy.RSIStrategy(buy_threshold=25, sell_threshold=75,\n"
        "                                 window=21, trade_size=100, max_position=500)\n"
        "    bt = strategy.Backtest(strat, prices=px, initial_cash=100_000).run()\n"
        "    results[ticker] = bt\n"
        "    print(f'{ticker}:  retorno = {bt.total_return:+.2%}  | '\n"
        "          f'trades = {bt.n_trades}  | '\n"
        "          f'equity final = R$ {bt.equity.iloc[-1]:,.0f}')\n"
    ),
    md(
        "## Q6 — Métricas vs IBOVESPA + buy-and-hold como referência\n\n"
        "Comparação que a v1 não fez: curvas rebaseadas a 100 e tabela com Sharpe, "
        "Sortino, Calmar, max drawdown, beta e alpha (Jensen) em base anualizada."
    ),
    code(
        "ibov_window = ibov_series.loc[WINDOW]\n"
        "bench_equity = rebase_to(ibov_window, base=100_000.0)\n"
        "\n"
        "curves = {'IBOV (BH)': rebase_to(ibov_window, base=100.0)}\n"
        "metrics = {}\n"
        "\n"
        "for ticker, bt in results.items():\n"
        "    eq_rebased = rebase_to(bt.equity, base=100.0)\n"
        "    curves[f'{ticker} RSI'] = eq_rebased\n"
        "    bh = strategy.buy_and_hold(prices[ticker].loc[WINDOW], initial_cash=100_000)\n"
        "    curves[f'{ticker} BH'] = rebase_to(bh, base=100.0)\n"
        "    metrics[f'{ticker} RSI'] = stats.performance_summary(\n"
        "        bt.equity, benchmark_equity=bench_equity)\n"
        "    metrics[f'{ticker} BH']  = stats.performance_summary(\n"
        "        bh, benchmark_equity=bench_equity)\n"
        "\n"
        "metrics['IBOV (BH)'] = stats.performance_summary(bench_equity, benchmark_equity=bench_equity)\n"
        "\n"
        "fig = plots.equity_curves(curves, title='RSI(21) vs Buy-and-Hold vs IBOV (base 100)')\n"
        "fig.show()\n"
        "\n"
        "pd.DataFrame(metrics).round(4)\n"
    ),
    md(
        "## Bonus — Duas melhorias implementadas\n\n"
        "1. **Filtro de tendência (SMA-200)** — RSI só compra se o preço estiver acima "
        "da média móvel de 200 dias. Evita comprar mean-reversion em downtrends.\n"
        "2. **Custos de corretagem** — parametrizáveis em bps; mostramos o impacto de "
        "10 bps por trade na curva de equity."
    ),
    code(
        "comparison = {}\n"
        "for ticker in ('BPAC11.SA', 'ITUB4.SA'):\n"
        "    px = prices[ticker].loc[WINDOW]\n"
        "    base = strategy.RSIStrategy(window=21)\n"
        "    trend = strategy.RSIStrategy(window=21, trend_filter_window=200)\n"
        "    costly = strategy.RSIStrategy(window=21, trend_filter_window=200,\n"
        "                                  commission_bps=10.0)\n"
        "    comparison[(ticker, 'RSI puro')] = strategy.Backtest(base, px).run().total_return\n"
        "    comparison[(ticker, 'RSI + SMA200')] = strategy.Backtest(trend, px).run().total_return\n"
        "    comparison[(ticker, 'RSI + SMA200 + 10bps')] = strategy.Backtest(costly, px).run().total_return\n"
        "\n"
        "print('Retorno total por estratégia:')\n"
        "for k, v in comparison.items():\n"
        "    print(f'  {k[0]:12s}  {k[1]:24s}  {v:+.2%}')\n"
    ),
    md(
        "---\n\n"
        "### Observações finais\n\n"
        "- **Reprodutibilidade**: este notebook depende exclusivamente de "
        "`safra_quant`, todas as funções com tipagem estrita e cobertas por testes "
        "determinísticos (`uv run pytest`).\n"
        "- **Performance vs benchmark**: nas duas ações testadas, a estratégia "
        "RSI(21) tende a underperformar o buy-and-hold em períodos de tendência "
        "forte (BPAC11 cresceu muito em 2019–2020). Trend-following + position "
        "sizing por volatilidade inversa seriam os próximos experimentos naturais.\n"
        "- **Notebook v1 preservado** em `archive/jptd_2022.ipynb` — útil para o "
        "diff didático com a v2."
    ),
]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = CELLS
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb["metadata"]["language_info"] = {"name": "python", "version": "3.11"}
    nbf.write(nb, NB_PATH)
    print(f"Wrote {NB_PATH}")


if __name__ == "__main__":
    main()
