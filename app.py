import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy.stats import norm
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
import warnings
from io import BytesIO
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Análise de Ativos B3",
    page_icon="📈",
    layout="wide"
)

# Funções principais
def get_historical_data(ticker, period="5y"):
    """Obtém dados históricos do ticker"""
    return yf.download(ticker, period=period)

def monte_carlo_simulation(data, simulations=1000, days=10):
    """Simulação de Monte Carlo baseada na variação histórica"""
    returns = data['Close'].pct_change().dropna()
    last_price = float(data['Close'].iloc[-1])  # Ensure float conversion
    
    # Parâmetros da distribuição
    mu = float(returns.mean())
    sigma = float(returns.std())
    
    # Simulação
    simulation_df = pd.DataFrame()
    for i in range(simulations):
        prices = [last_price]
        for _ in range(days):
            # Movimento browniano geométrico
            drift = mu - (0.5 * sigma**2)
            shock = sigma * np.random.normal()
            price = prices[-1] * np.exp(drift + shock)
            prices.append(float(price))  # Ensure float conversion
        simulation_df[f'Sim_{i}'] = prices[1:]
    
    return simulation_df.astype(float)  # Ensure all columns are float

def calculate_confidence_intervals(data, days=10):
    """Calcula intervalos de confiança usando ARIMA-GARCH"""
    returns = data['Close'].pct_change().dropna() * 100
    
    # Modelo ARIMA(1,1,1)
    model = ARIMA(returns, order=(1,1,1))
    model_fit = model.fit()
    
    # Modelo GARCH(1,1)
    garch = arch_model(model_fit.resid, vol='Garch', p=1, q=1)
    garch_fit = garch.fit(disp='off')
    
    # Previsão
    forecast = garch_fit.forecast(horizon=days)
    conf_intervals = []
    for i in range(days):
        lower = norm.ppf(0.025, loc=0, scale=np.sqrt(forecast.variance.values[-1,i]))
        upper = norm.ppf(0.975, loc=0, scale=np.sqrt(forecast.variance.values[-1,i]))
        conf_intervals.append((lower, upper))
    
    return conf_intervals

def plot_simulation_results(simulations, conf_intervals):
    """Plota os resultados da simulação"""
    fig, ax = plt.subplots(figsize=(12,6))
    
    # Plot simulations
    for col in simulations.columns:
        ax.plot(simulations[col], color='blue', alpha=0.05)
    
    # Calculate and plot mean
    mean_sim = simulations.mean(axis=1)
    ax.plot(mean_sim, color='red', linewidth=2, label='Média')
    
    # Calculate and plot confidence intervals
    lower_bound = simulations.quantile(0.025, axis=1)
    upper_bound = simulations.quantile(0.975, axis=1)
    ax.fill_between(simulations.index, lower_bound, upper_bound, color='orange', alpha=0.3, label='Intervalo 95%')
    
    ax.set_title('Simulação de Monte Carlo - Preços Futuros')
    ax.set_xlabel('Dias')
    ax.set_ylabel('Preço')
    ax.legend()
    st.pyplot(fig)

# Interface Streamlit
st.title("Análise e Previsão de Ativos B3")

# Sidebar - Inputs do usuário
st.sidebar.header("Parâmetros")
ticker = st.sidebar.text_input("Digite o ticker do ativo (ex: BOVA11.SA)", "BOVA11.SA")
period = st.sidebar.selectbox("Período histórico", ["1y", "3y", "5y", "max"], index=2)
simulations = st.sidebar.slider("Número de simulações", 100, 10000, 1000)
confidence_level = st.sidebar.slider("Nível de confiança (%)", 90, 99, 95)

# Obter dados
if ticker:
    with st.spinner(f"Carregando dados para {ticker}..."):
        data = get_historical_data(ticker, period)
        
        if data.empty:
            st.error("Não foi possível obter dados para este ticker. Verifique o código e tente novamente.")
        else:
            # Mostrar dados históricos
            st.header(f"Dados Históricos - {ticker}")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Últimos 5 pregões:")
                st.dataframe(data.tail())
            with col2:
                st.write("Estatísticas descritivas:")
                st.dataframe(data.describe())
            
            # Gráfico de preços
            st.subheader("Série Histórica de Preços")
            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(data['Close'], label='Preço de Fechamento')
            ax.plot(data['High'], label='Máxima', alpha=0.3)
            ax.plot(data['Low'], label='Mínima', alpha=0.3)
            ax.set_title(f"Histórico de Preços - {ticker}")
            ax.legend()
            st.pyplot(fig)
            
            # Volume
            st.subheader("Volume Negociado")
            st.bar_chart(data['Volume'])
            
            # Simulação de Monte Carlo
            st.header("Simulação de Preços Futuros (10 dias)")
            simulations_df = monte_carlo_simulation(data, simulations, 10)
            conf_intervals = calculate_confidence_intervals(data)
            
            # Resultados da simulação
            plot_simulation_results(simulations_df, conf_intervals)
            
            # Estatísticas
            st.subheader("Estatísticas da Simulação")
            last_price = float(data['Close'].iloc[-1])
            final_prices = simulations_df.iloc[-1]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Preço Atual", f"R$ {last_price:.2f}")
            with col2:
                mean_price = final_prices.mean()
                st.metric("Preço Médio Projetado", f"R$ {mean_price:.2f}")
            with col3:
                std_dev = final_prices.std()
                st.metric("Desvio Padrão", f"R$ {std_dev:.2f}")
            
            # Probabilidades
            st.subheader("Probabilidades")
            price_input = st.number_input("Digite um preço para análise:", value=float(last_price))
            prob_above = (final_prices > price_input).mean() * 100
            prob_below = (final_prices < price_input).mean() * 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"Prob. preço > R$ {price_input:.2f}", f"{prob_above:.1f}%")
            with col2:
                st.metric(f"Prob. preço < R$ {price_input:.2f}", f"{prob_below:.1f}%")
            
            # Comparação com Ibovespa
            st.header("Comparação com Ibovespa")
            if ticker != "^BVSP":
                with st.spinner("Carregando dados do Ibovespa..."):
                    ibov = get_historical_data("^BVSP", period)
                    if not ibov.empty:
                        # Align data by date before calculating correlation
                        aligned_data = pd.merge(data['Close'], ibov['Close'], 
                                             left_index=True, right_index=True,
                                             how='inner')
                        correlation = aligned_data.iloc[:,0].corr(aligned_data.iloc[:,1])
                        st.write(f"Correlação com Ibovespa: {correlation:.2f}")
                        
                        fig, ax = plt.subplots(figsize=(12,6))
                        ax.plot(data['Close']/data['Close'].iloc[0], label=ticker)
                        ax.plot(ibov['Close']/ibov['Close'].iloc[0], label='Ibovespa')
                        ax.set_title("Desempenho Relativo")
                        ax.legend()
                        st.pyplot(fig)
            
            # Download dos resultados
            st.header("Exportar Resultados")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Gerar Relatório em Excel"):
                    report = pd.DataFrame({
                        'Métrica': ['Preço Atual', 'Preço Médio Projetado', 'Desvio Padrão', 
                                   f'Prob. > R$ {price_input:.2f}', f'Prob. < R$ {price_input:.2f}'],
                        'Valor': [last_price, mean_price, std_dev, prob_above, prob_below]
                    })
                    # Create Excel file in memory
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        report.to_excel(writer, index=False, sheet_name='Relatório')
                        writer.close()
                    
                    st.download_button(
                        label="Baixar Excel",
                        data=output.getvalue(),
                        file_name=f"relatorio_{ticker}.xlsx",
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )
            
            with col2:
                if st.button("Gerar Relatório em PDF"):
                    from fpdf import FPDF
                    import tempfile
                    import os
                    
                    # Create PDF
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Arial", size=12)
                    
                    # Title
                    pdf.set_font("Arial", 'B', 16)
                    pdf.cell(200, 10, txt=f"Relatório Completo - {ticker}", ln=1, align='C')
                    pdf.ln(15)
                    pdf.set_font("Arial", size=12)
                    
                    # Section 1: Basic Info
                    pdf.set_font("Arial", 'B', 14)
                    pdf.cell(200, 10, txt="Informações Básicas", ln=1)
                    pdf.set_font("Arial", size=12)
                    pdf.cell(200, 10, txt=f"Período: {period}", ln=1)
                    pdf.cell(200, 10, txt=f"Preço Atual: R$ {last_price:.2f}", ln=1)
                    pdf.cell(200, 10, txt=f"Preço Médio Projetado: R$ {mean_price:.2f}", ln=1)
                    pdf.cell(200, 10, txt=f"Desvio Padrão: R$ {std_dev:.2f}", ln=1)
                    pdf.cell(200, 10, txt=f"Prob. > R$ {price_input:.2f}: {prob_above:.1f}%", ln=1)
                    pdf.cell(200, 10, txt=f"Prob. < R$ {price_input:.2f}: {prob_below:.1f}%", ln=1)
                    pdf.ln(10)
                    
                    # Section 2: Historical Price Chart
                    pdf.set_font("Arial", 'B', 14)
                    pdf.cell(200, 10, txt="Gráfico Histórico de Preços", ln=1)
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
                        fig, ax = plt.subplots(figsize=(12,6))
                        ax.plot(data['Close'], label='Preço de Fechamento')
                        ax.plot(data['High'], label='Máxima', alpha=0.3)
                        ax.plot(data['Low'], label='Mínima', alpha=0.3)
                        ax.set_title(f"Histórico de Preços - {ticker}")
                        ax.legend()
                        fig.savefig(tmpfile.name, bbox_inches='tight', dpi=100)
                        plt.close(fig)
                        pdf.image(tmpfile.name, x=10, w=190)
                        os.unlink(tmpfile.name)
                    pdf.ln(5)
                    
                    # Section 3: Volume Chart
                    pdf.set_font("Arial", 'B', 14)
                    pdf.cell(200, 10, txt="Volume Negociado", ln=1)
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
                        fig, ax = plt.subplots(figsize=(12,4))
                        
                        # Convert dates to matplotlib numeric format
                        dates = plt.dates.date2num(data.index.to_pydatetime())
                        volumes = data['Volume'].values
                        
                        # Calculate bar width (1 day width)
                        width = 1.0
                        
                        # Create bar plot
                        bars = ax.bar(dates, volumes, width=width)
                        
                        # Format x-axis with proper dates
                        ax.xaxis_date()
                        ax.xaxis.set_major_formatter(plt.dates.DateFormatter('%Y-%m-%d'))
                        fig.autofmt_xdate()
                        ax.set_title(f"Volume - {ticker}")
                        
                        fig.savefig(tmpfile.name, bbox_inches='tight', dpi=100)
                        plt.close(fig)
                        pdf.image(tmpfile.name, x=10, w=190)
                        os.unlink(tmpfile.name)
                    pdf.ln(5)
                    
                    # Section 4: Monte Carlo Simulation
                    pdf.set_font("Arial", 'B', 14)
                    pdf.cell(200, 10, txt="Simulação de Monte Carlo", ln=1)
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
                        fig, ax = plt.subplots(figsize=(12,6))
                        for col in simulations_df.columns:
                            ax.plot(simulations_df[col], color='blue', alpha=0.05)
                        ax.plot(simulations_df.mean(axis=1), color='red', linewidth=2, label='Média')
                        lower = simulations_df.quantile(0.025, axis=1)
                        upper = simulations_df.quantile(0.975, axis=1)
                        ax.fill_between(simulations_df.index, lower, upper, color='orange', alpha=0.3, label='Intervalo 95%')
                        ax.set_title('Simulação de Monte Carlo - Preços Futuros')
                        ax.legend()
                        fig.savefig(tmpfile.name, bbox_inches='tight', dpi=100)
                        plt.close(fig)
                        pdf.image(tmpfile.name, x=10, w=190)
                        os.unlink(tmpfile.name)
                    
                    # Save to buffer
                    pdf_output = BytesIO()
                    pdf_output.write(pdf.output(dest='S').encode('latin1'))
                    
                    st.download_button(
                        label="Baixar PDF Completo",
                        data=pdf_output.getvalue(),
                        file_name=f"relatorio_completo_{ticker}.pdf",
                        mime='application/pdf'
                    )

# Rodapé
st.sidebar.markdown("---")
st.sidebar.markdown(f"Última atualização: {pd.Timestamp.now().strftime('%d/%m/%Y %H:%M')}")
