#!/usr/bin/env python3
"""
Jamaica Government Securities Yield Curve Analytics Platform
Professional yield curve visualization for Government of Jamaica securities
"""

import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from pathlib import Path
import glob
import logging
from datetime import datetime
import re

# Configuration
OUTPUT_DIR = Path("output")
APP_TITLE = "Jamaica Government Securities"
APP_SUBTITLE = "Yield Curve Analytics"

# Professional color palette for financial applications
COLORS = {
    'primary': '#1E3A5F',      # Deep financial blue
    'secondary': '#2E4C6D',    # Medium blue
    'accent': '#C8102E',       # Jamaica red accent
    'text_primary': '#1A1A1A', # Near black
    'text_secondary': '#6C757D', # Gray
    'background': '#FFFFFF',   # White
    'grid': '#E9ECEF',        # Light gray
    'treasury': '#1E3A5F',    # Deep blue for treasury
    'money_market': '#C8102E'  # Red for money market
}

# Setup logging
logging.basicConfig(level=logging.INFO)

class GovernmentSecuritiesAnalytics:
    def __init__(self):
        """Initialize the Government Securities Analytics platform"""
        self.app = dash.Dash(__name__, title=f"{APP_TITLE} - {APP_SUBTITLE}")
        self.setup_layout()
        self.setup_callbacks()
    
    def find_latest_csv(self) -> str:
        """Find the latest combined securities CSV file"""
        if not OUTPUT_DIR.exists():
            raise ValueError(f"Data directory {OUTPUT_DIR} not found. Execute data processing pipeline.")
        
        pattern = str(OUTPUT_DIR / "boj_securities_*.csv")
        csv_files = glob.glob(pattern)
        
        if not csv_files:
            raise ValueError(f"No securities data found in {OUTPUT_DIR}. Execute data processing pipeline.")
        
        latest_file = max(csv_files)
        logging.info(f"Loading securities data: {latest_file}")
        return latest_file
    
    def load_and_prepare_data(self) -> pd.DataFrame:
        """Load and prepare securities data for analysis"""
        csv_file = self.find_latest_csv()
        df = pd.read_csv(csv_file)
        
        # Filter out invalid yields
        df = df[df['yield'] > 0].copy()
        
        # Prepare term structure data
        df['term_to_maturity'] = df.apply(self._calculate_term_structure, axis=1)
        df['maturity_classification'] = df.apply(self._classify_maturity, axis=1)
        df['security_classification'] = df.apply(self._classify_security, axis=1)
        df['hover_data'] = df.apply(self._format_hover_data, axis=1)
        
        # Sort by term to maturity for yield curve construction
        df = df.sort_values('term_to_maturity')
        
        logging.info(f"Processed {len(df)} securities for yield curve analysis")
        return df
    
    def _calculate_term_structure(self, row) -> float:
        """Calculate standardized term to maturity for yield curve plotting"""
        if row['security_type'] == 'Treasury Bill' and pd.notna(row['tenure_months']):
            return row['tenure_months'] / 12  # Convert to years for standardization
        else:
            return row['time_to_maturity_years']
    
    def _classify_maturity(self, row) -> str:
        """Classify securities by maturity bucket"""
        years = self._calculate_term_structure(row)
        if years <= 1:
            return "Short-term (≤1Y)"
        elif years <= 5:
            return "Medium-term (1-5Y)"
        else:
            return "Long-term (>5Y)"
    
    def _classify_security(self, row) -> str:
        """Standardize security type classification"""
        if row['security_type'] == 'Treasury Bill':
            return 'Treasury Bills'
        else:
            return 'Benchmark Investment Notes'
    
    def _format_hover_data(self, row) -> str:
        """Format professional hover information"""
        term_years = self._calculate_term_structure(row)
        
        # Format term display
        if row['security_type'] == 'Treasury Bill':
            term_display = f"{int(row['tenure_months'])}M Treasury Bill"
        else:
            term_display = f"{term_years:.1f}Y Benchmark Note"
        
        # Build hover information
        hover_components = [
            f"<b>{term_display}</b>",
            f"<b>Yield:</b> {row['yield']:.2f}%",
            f"<b>Issue Date:</b> {row['issue_date']}",
            f"<b>Maturity:</b> {row['maturity_date']}",
            f"<b>Term:</b> {term_years:.2f} years"
        ]
        
        if pd.notna(row['fixed_rate']):
            hover_components.insert(2, f"<b>Coupon Rate:</b> {row['fixed_rate']:.2f}%")
        
        # Add source reference
        source_ref = row['source_title'][:60] + "..." if len(row['source_title']) > 60 else row['source_title']
        hover_components.append(f"<b>Source:</b> {source_ref}")
        
        return "<br>".join(hover_components)
    
    def calculate_yield_curve_metrics(self, df: pd.DataFrame) -> dict:
        """Calculate key yield curve metrics for display"""
        if df.empty:
            return {}
        
        metrics = {}
        
        # Yield spread analysis
        if len(df) >= 2:
            short_yields = df[df['term_to_maturity'] <= 1]['yield']
            long_yields = df[df['term_to_maturity'] >= 5]['yield']
            
            if not short_yields.empty and not long_yields.empty:
                metrics['yield_spread'] = long_yields.max() - short_yields.min()
        
        # Curve characteristics
        metrics['securities_count'] = len(df)
        metrics['yield_range'] = f"{df['yield'].min():.2f}% - {df['yield'].max():.2f}%"
        metrics['avg_yield'] = df['yield'].mean()
        
        # Maturity distribution
        treasury_count = len(df[df['security_classification'] == 'Treasury Bills'])
        bin_count = len(df[df['security_classification'] == 'Benchmark Investment Notes'])
        
        metrics['treasury_bills'] = treasury_count
        metrics['benchmark_notes'] = bin_count
        
        return metrics
    
    def create_yield_curve_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create professional yield curve visualization"""
        if df.empty:
            return self._create_no_data_chart()
        
        # Separate security types
        treasury_df = df[df['security_classification'] == 'Treasury Bills'].copy()
        bin_df = df[df['security_classification'] == 'Benchmark Investment Notes'].copy()
        
        fig = go.Figure()
        
        # Treasury Bills
        if not treasury_df.empty:
            fig.add_trace(go.Scatter(
                x=treasury_df['term_to_maturity'],
                y=treasury_df['yield'],
                mode='markers+lines',
                name='Treasury Bills',
                text=treasury_df['hover_data'],
                hovertemplate='%{text}<extra></extra>',
                marker=dict(
                    size=14,
                    color=COLORS['treasury'],
                    symbol='circle',
                    line=dict(width=2, color='white')
                ),
                line=dict(
                    color=COLORS['treasury'],
                    width=3,
                    dash='solid'
                )
            ))
        
        # Benchmark Investment Notes (BINs)
        if not bin_df.empty:
            fig.add_trace(go.Scatter(
                x=bin_df['term_to_maturity'],
                y=bin_df['yield'],
                mode='markers+lines',
                name='Benchmark Investment Notes',
                text=bin_df['hover_data'],
                hovertemplate='%{text}<extra></extra>',
                marker=dict(
                    size=14,
                    color=COLORS['money_market'],
                    symbol='diamond',
                    line=dict(width=2, color='white')
                ),
                line=dict(
                    color=COLORS['money_market'],
                    width=3,
                    dash='solid'
                )
            ))
        
        # Professional styling
        fig.update_layout(
            title=dict(
                text=f"<b>Government of Jamaica Securities Yield Curve</b><br><span style='font-size:16px; color:{COLORS['text_secondary']}'>Term Structure of Interest Rates</span>",
                font=dict(size=24, color=COLORS['text_primary'], family="Segoe UI, Helvetica Neue, Arial"),
                x=0.5,
                y=0.95
            ),
            xaxis=dict(
                title="<b>Term to Maturity (Years)</b>",
                titlefont=dict(size=16, color=COLORS['text_primary'], family="Segoe UI"),
                tickfont=dict(size=14, color=COLORS['text_primary'], family="Segoe UI"),
                gridcolor=COLORS['grid'],
                showgrid=True,
                zeroline=False,
                tickformat='.1f'
            ),
            yaxis=dict(
                title="<b>Yield to Maturity (%)</b>",
                titlefont=dict(size=16, color=COLORS['text_primary'], family="Segoe UI"),
                tickfont=dict(size=14, color=COLORS['text_primary'], family="Segoe UI"),
                gridcolor=COLORS['grid'],
                showgrid=True,
                zeroline=False,
                tickformat='.2f'
            ),
            legend=dict(
                font=dict(size=14, family="Segoe UI"),
                x=0.02,
                y=0.98,
                bgcolor='rgba(248, 249, 250, 0.9)',
                bordercolor=COLORS['grid'],
                borderwidth=1
            ),
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['background'],
            hovermode='closest',
            font=dict(family="Segoe UI, Helvetica Neue, Arial"),
            margin=dict(l=80, r=80, t=120, b=80),
            height=650
        )
        
        return fig
    
    def _create_no_data_chart(self) -> go.Figure:
        """Create chart for no data scenario"""
        fig = go.Figure()
        fig.add_annotation(
            text="<b>No Securities Data Available</b><br><span style='color:#6C757D'>Execute data collection pipeline to populate yield curve</span>",
            xref="paper", yref="paper",
            x=0.5, y=0.5, xanchor='center', yanchor='middle',
            font=dict(size=18, color=COLORS['text_secondary'], family="Segoe UI")
        )
        fig.update_layout(
            title="Government of Jamaica Securities Yield Curve",
            xaxis_title="Term to Maturity (Years)",
            yaxis_title="Yield to Maturity (%)",
            font=dict(size=16, family="Segoe UI"),
            plot_bgcolor=COLORS['background'],
            paper_bgcolor=COLORS['background']
        )
        return fig
    
    def setup_layout(self):
        """Setup professional dashboard layout"""
        self.app.layout = html.Div([
            # Header section
            html.Div([
                html.Div([
                    html.H1(
                        APP_TITLE,
                        style={
                            'margin': '0',
                            'color': COLORS['primary'],
                            'fontSize': '32px',
                            'fontWeight': '700',
                            'fontFamily': 'Segoe UI, Helvetica Neue, Arial',
                            'letterSpacing': '-0.5px'
                        }
                    ),
                    html.H2(
                        APP_SUBTITLE,
                        style={
                            'margin': '5px 0 0 0',
                            'color': COLORS['text_secondary'],
                            'fontSize': '18px',
                            'fontWeight': '400',
                            'fontFamily': 'Segoe UI, Helvetica Neue, Arial'
                        }
                    ),
                ], style={'flex': '1'}),
                
                html.Div([
                    html.Button(
                        'Refresh Data',
                        id='refresh-button',
                        n_clicks=0,
                        style={
                            'fontSize': '14px',
                            'fontWeight': '600',
                            'padding': '12px 24px',
                            'backgroundColor': COLORS['primary'],
                            'color': 'white',
                            'border': 'none',
                            'borderRadius': '6px',
                            'cursor': 'pointer',
                            'fontFamily': 'Segoe UI, Helvetica Neue, Arial',
                            'transition': 'all 0.2s'
                        }
                    ),
                ], style={'display': 'flex', 'alignItems': 'center'})
            ], style={
                'display': 'flex',
                'justifyContent': 'space-between',
                'alignItems': 'flex-start',
                'padding': '30px 0 20px 0',
                'borderBottom': f'1px solid {COLORS["grid"]}'
            }),
            
            # Metrics dashboard
            html.Div(id='metrics-dashboard', style={'margin': '20px 0'}),
            
            # Main yield curve chart
            dcc.Graph(
                id='yield-curve-chart',
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d', 'autoScale2d']
                }
            ),
            
            # Footer information
            html.Div([
                html.Div([
                    html.Span("Last Updated: ", style={'fontWeight': '600', 'color': COLORS['text_secondary']}),
                    html.Span(id='last-updated-time', style={'color': COLORS['text_secondary']})
                ], style={'marginBottom': '10px'}),
                html.Div(
                    id='data-summary',
                    style={'color': COLORS['text_secondary'], 'fontSize': '14px'}
                )
            ], style={
                'textAlign': 'center',
                'marginTop': '30px',
                'padding': '20px',
                'backgroundColor': '#F8F9FA',
                'borderRadius': '8px',
                'fontFamily': 'Segoe UI, Helvetica Neue, Arial',
                'fontSize': '14px'
            })
        ], style={
            'maxWidth': '1400px',
            'margin': '0 auto',
            'padding': '0 40px 40px 40px',
            'fontFamily': 'Segoe UI, Helvetica Neue, Arial',
            'backgroundColor': COLORS['background'],
            'minHeight': '100vh'
        })
    
    def create_metrics_dashboard(self, metrics: dict) -> html.Div:
        """Create key metrics dashboard"""
        if not metrics:
            return html.Div()
        
        metric_cards = []
        
        # Securities count
        metric_cards.append(
            html.Div([
                html.Div("Securities Analyzed", className='metric-label'),
                html.Div(str(metrics.get('securities_count', 0)), className='metric-value'),
                html.Div(f"TB: {metrics.get('treasury_bills', 0)} | BIN: {metrics.get('benchmark_notes', 0)}", className='metric-detail')
            ], className='metric-card')
        )
        
        # Yield range
        metric_cards.append(
            html.Div([
                html.Div("Yield Range", className='metric-label'),
                html.Div(metrics.get('yield_range', 'N/A'), className='metric-value'),
                html.Div(f"Average: {metrics.get('avg_yield', 0):.2f}%", className='metric-detail')
            ], className='metric-card')
        )
        
        # Yield spread if available
        if 'yield_spread' in metrics:
            metric_cards.append(
                html.Div([
                    html.Div("Yield Spread", className='metric-label'),
                    html.Div(f"{metrics['yield_spread']:.0f} bps", className='metric-value'),
                    html.Div("Long-Short Differential", className='metric-detail')
                ], className='metric-card')
            )
        
        return html.Div(
            metric_cards,
            style={
                'display': 'grid',
                'gridTemplateColumns': 'repeat(auto-fit, minmax(200px, 1fr))',
                'gap': '20px',
                'margin': '20px 0'
            }
        )
    
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        @self.app.callback(
            [Output('yield-curve-chart', 'figure'),
             Output('last-updated-time', 'children'),
             Output('data-summary', 'children'),
             Output('metrics-dashboard', 'children')],
            [Input('refresh-button', 'n_clicks')]
        )
        def update_analytics_dashboard(n_clicks):
            try:
                df = self.load_and_prepare_data()
                fig = self.create_yield_curve_chart(df)
                metrics = self.calculate_yield_curve_metrics(df)
                
                # Update timestamp
                last_updated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # Create summary
                treasury_count = metrics.get('treasury_bills', 0)
                bin_count = metrics.get('benchmark_notes', 0)
                data_summary = f"Yield curve constructed from {treasury_count} Treasury Bills and {bin_count} Benchmark Investment Notes"
                
                # Create metrics dashboard
                metrics_dashboard = self.create_metrics_dashboard(metrics)
                
                return fig, last_updated, data_summary, metrics_dashboard
                
            except Exception as e:
                logging.error(f"Analytics update error: {e}")
                
                error_fig = self._create_no_data_chart()
                error_summary = f"Data processing error: {str(e)}"
                
                return error_fig, "Error", error_summary, html.Div()
    
    def run(self, debug=False, port=8050):
        """Launch the analytics platform"""
        print(f"\n📊 Launching {APP_TITLE} {APP_SUBTITLE}")
        print(f"🌐 Platform URL: http://localhost:{port}")
        print(f"📁 Data source: {OUTPUT_DIR.absolute()}")
        print("🔄 Use 'Refresh Data' to reload latest market data\n")
        
        self.app.run_server(debug=debug, port=port, host='0.0.0.0')


# Add CSS styling for metric cards
def inject_css():
    return """
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #E9ECEF;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-label {
        font-size: 14px;
        color: #6C757D;
        font-weight: 600;
        margin-bottom: 8px;
        font-family: "Segoe UI", "Helvetica Neue", Arial;
    }
    .metric-value {
        font-size: 24px;
        color: #1E3A5F;
        font-weight: 700;
        margin-bottom: 4px;
        font-family: "Segoe UI", "Helvetica Neue", Arial;
    }
    .metric-detail {
        font-size: 12px;
        color: #6C757D;
        font-family: "Segoe UI", "Helvetica Neue", Arial;
    }
    """


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Jamaica Government Securities Yield Curve Analytics')
    parser.add_argument('--port', type=int, default=8050, help='Application port')
    parser.add_argument('--debug', action='store_true', help='Development mode')
    
    args = parser.parse_args()
    
    try:
        # Inject CSS
        app_css = inject_css()
        
        platform = GovernmentSecuritiesAnalytics()
        platform.run(debug=args.debug, port=args.port)
    except Exception as e:
        print(f"❌ Platform initialization failed: {e}")
        print("📋 Prerequisites:")
        print("   1. Execute data collection: python boj_scraper.py")
        print("   2. Process securities data: python boj_processor.py")
        print("   3. Install dependencies: pip install dash plotly pandas")


if __name__ == "__main__":
    main()