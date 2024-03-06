import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np


def simulate_portfolio(assets, allocations, start_date, end_date, initial_investment=0, monthly_investment=0):
    # Obtém os preços ajustados de cada ativo
    prices = pd.DataFrame({asset: yf.download(asset, start=start_date, end=end_date)['Adj Close'] for asset in assets})

    # Calcula o retorno diário de cada ativo
    daily_returns = prices.pct_change()

    # Remove o primeiro dia, que pode conter valores nulos após o cálculo do retorno
    daily_returns = daily_returns.dropna()

    # Adiciona o investimento inicial ao primeiro dia
    portfolio_value = initial_investment + np.sum(daily_returns.iloc[0] * allocations * initial_investment)

    # Lista para armazenar o valor do portfólio em cada data
    portfolio_values = [portfolio_value]

    # Loop para simular o investimento mensal e calcular o valor do portfólio em cada data
    for date, daily_return in daily_returns.iloc[1:].iterrows():
        # Calcula o valor do portfólio após o aporte mensal
        if date.day == 1:  # Verifica se é o primeiro dia do mês
            portfolio_value += monthly_investment

        # Atualiza o valor do portfólio para cada ativo com base nos retornos diários e alocações
        portfolio_value *= (1 + np.sum(daily_return * allocations))

        # Adiciona o valor do portfólio na lista
        portfolio_values.append(portfolio_value)

    # Converte a lista para um DataFrame de pandas
    portfolio_cumulative_returns = pd.Series(portfolio_values, index=daily_returns.index)

    return portfolio_cumulative_returns


# Definindo os ativos, alocações e parâmetros adicionais
assets = ['TAEE11.SA', 'ITSA4.SA', 'BBAS3.SA', 'PETR4.SA', 'TGMA3.SA', 'AGRO3.SA', 'VULC3.SA', 'CMIG4.SA', 'USIM5.SA',
          'KLBN11.SA']  # Ativos (Apple, Microsoft, Google, Amazon)
allocations = [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]  # Alocações (10% em cada ativo)
start_date = '2018-07-30'
end_date = '2023-07-30'
initial_investment = 1000  # Investimento inicial de R$ 10.000
monthly_investment = 200  # Aporte mensal de R$ 1.000

# Simulando a carteira de investimentos
portfolio_returns = simulate_portfolio(assets, allocations, start_date, end_date, initial_investment, monthly_investment)

# Obtendo os dados históricos do IBOV
ibov_data = yf.download('^BVSP', start=start_date, end=end_date)['Adj Close']
ibov_daily_returns = ibov_data.pct_change().dropna()

# Simulando o investimento inicial e aportes mensais no IBOV
ibov_investment_values = [initial_investment]

for i in range(1, len(ibov_daily_returns)):
    if ibov_daily_returns.index[i].day == 1:  # Verifica se é o primeiro dia do mês
        ibov_investment_values.append(ibov_investment_values[-1] + monthly_investment)
    else:
        ibov_investment_values.append(ibov_investment_values[-1])

    ibov_investment_values[-1] *= (1 + ibov_daily_returns[i])

ibov_cumulative_returns = pd.Series(ibov_investment_values, index=ibov_daily_returns.index)

# Plota os resultados
plt.figure(figsize=(12, 8))  # Reduzindo a altura da figura para acomodar os dois subplots

# Calculating the portfolio returns without monthly investments
portfolio_returns_no_investments = simulate_portfolio(assets, allocations, start_date, end_date, initial_investment, 0)

# Calculating the IBOV returns without monthly investments
ibov_cumulative_returns_no_investments = pd.Series(initial_investment * (1 + ibov_daily_returns).cumprod(), index=ibov_daily_returns.index)

# Gráfico da evolução do valor do portfólio e do valor do patrimônio investindo apenas no IBOV
plt.subplot(2, 1, 1)  # Modificando para criar 2 subplots em uma coluna
plt.plot(portfolio_returns, label='Carteira Simulada com Aportes')
plt.plot(portfolio_returns_no_investments, label='Carteira Simulada sem Aportes')
plt.plot(ibov_cumulative_returns, label='IBOV com Aportes')
plt.plot(ibov_cumulative_returns_no_investments, label='IBOV sem Aportes')
plt.axhline(y=initial_investment, color='r', linestyle='--', label='Investimento Inicial')
plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))
plt.text(0, initial_investment, f'Investimento Inicial: R$ {initial_investment:,.0f}', color='r', ha='left', va='center')
plt.legend()
plt.ylabel('Valor (R$)')
plt.title('Evolução do Valor do Portfólio e do Patrimônio Investindo no IBOV')

# Gráfico do desempenho da carteira e do IBOV em relação ao investimento inicial
plt.subplot(2, 1, 2)  # Modificando para criar 2 subplots em uma coluna
portfolio_relative_performance = ((portfolio_returns - initial_investment) / initial_investment) * 100
portfolio_relative_performance_no_investments = ((portfolio_returns_no_investments - initial_investment) / initial_investment) * 100
ibov_relative_performance = ((ibov_cumulative_returns - initial_investment) / initial_investment) * 100
ibov_relative_performance_no_investments = ((ibov_cumulative_returns_no_investments - initial_investment) / initial_investment) * 100

# Calculate the maximum value between the portfolio and IBOV relative performance
max_relative_performance = max(
    portfolio_relative_performance.max(),
    ibov_relative_performance.max(),
    portfolio_relative_performance_no_investments.max(),
    ibov_relative_performance_no_investments.max()
)

# Set the y-axis limits from the minimum value to the maximum value
plt.ylim(portfolio_relative_performance.min(), max_relative_performance)

plt.plot(portfolio_relative_performance, label='Carteira Simulada com Aportes')
plt.plot(portfolio_relative_performance_no_investments, label='Carteira Simulada sem Aportes')
plt.plot(ibov_relative_performance, label='IBOV com Aportes')
plt.plot(ibov_relative_performance_no_investments, label='IBOV sem Aportes')
plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: '{:.1f}%'.format(x)))
plt.legend()
plt.ylabel('(%)')
plt.title('Desempenho da Carteira e do IBOV em relação ao Investimento Inicial')

plt.show()
