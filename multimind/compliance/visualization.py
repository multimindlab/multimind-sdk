"""
Visualization tools for compliance monitoring and evaluation.
Provides interactive dashboards and plots for monitoring model compliance.
"""

from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import webbrowser
from threading import Timer

class ComplianceVisualizer:
    """Visualization tools for compliance monitoring."""
    
    def __init__(self, results_path: Optional[str] = None):
        self.results_path = results_path
        self.metrics_history: List[Dict[str, Any]] = []
        if results_path:
            self.load_results(results_path)

    def load_results(self, path: str) -> None:
        """Load training results from file."""
        with open(path, 'r') as f:
            data = json.load(f)
            self.metrics_history = data['training_results']['metrics_history']

    def plot_metrics_history(
        self,
        metrics: List[str] = None,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Plot compliance metrics history."""
        if metrics is None:
            metrics = ['bias', 'privacy', 'transparency', 'fairness']
        
        df = pd.DataFrame(self.metrics_history)
        
        fig = make_subplots(
            rows=len(metrics),
            cols=1,
            subplot_titles=[f"{m.capitalize()} Score" for m in metrics]
        )
        
        for i, metric in enumerate(metrics, 1):
            fig.add_trace(
                go.Scatter(
                    y=df[f"{metric}_score"],
                    name=metric.capitalize(),
                    mode='lines+markers'
                ),
                row=i,
                col=1
            )
        
        fig.update_layout(
            height=300 * len(metrics),
            title_text="Compliance Metrics History",
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig

    def plot_violations_heatmap(
        self,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Plot violations heatmap."""
        violations = []
        for metrics in self.metrics_history:
            for metric in ['bias', 'privacy', 'transparency', 'fairness']:
                violations.append({
                    'metric': metric,
                    'timestamp': metrics['timestamp'],
                    'value': getattr(metrics, f"{metric}_score")
                })
        
        df = pd.DataFrame(violations)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        
        pivot = df.pivot_table(
            values='value',
            index='hour',
            columns='metric',
            aggfunc='mean'
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='RdYlGn_r'
        ))
        
        fig.update_layout(
            title="Compliance Violations Heatmap",
            xaxis_title="Metric",
            yaxis_title="Hour of Day"
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig

    def create_dashboard(
        self,
        port: int = 8050,
        debug: bool = False
    ) -> None:
        """Create interactive dashboard for compliance monitoring."""
        app = dash.Dash(__name__)
        
        app.layout = html.Div([
            html.H1("Compliance Monitoring Dashboard"),
            
            dcc.Tabs([
                dcc.Tab(label='Metrics History', children=[
                    dcc.Graph(id='metrics-history')
                ]),
                dcc.Tab(label='Violations Heatmap', children=[
                    dcc.Graph(id='violations-heatmap')
                ]),
                dcc.Tab(label='Recommendations', children=[
                    html.Div(id='recommendations')
                ])
            ]),
            
            dcc.Interval(
                id='interval-component',
                interval=5*1000,  # Update every 5 seconds
                n_intervals=0
            )
        ])
        
        @app.callback(
            [Output('metrics-history', 'figure'),
             Output('violations-heatmap', 'figure'),
             Output('recommendations', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_graphs(n):
            metrics_fig = self.plot_metrics_history()
            heatmap_fig = self.plot_violations_heatmap()
            
            recommendations = []
            for metrics in self.metrics_history[-5:]:  # Last 5 recommendations
                for metric in ['bias', 'privacy', 'transparency', 'fairness']:
                    score = getattr(metrics, f"{metric}_score")
                    if score < 0.8:  # Threshold
                        recommendations.append(
                            html.Div([
                                html.H4(f"{metric.capitalize()} Alert"),
                                html.P(f"Score: {score:.2f}"),
                                html.P(f"Time: {metrics['timestamp']}")
                            ])
                        )
            
            return metrics_fig, heatmap_fig, recommendations
        
        def open_browser():
            webbrowser.open_new(f'http://localhost:{port}/')
        
        Timer(1, open_browser).start()
        app.run_server(debug=debug, port=port)

    def plot_compliance_radar(
        self,
        metrics: Dict[str, float],
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Plot compliance metrics on a radar chart."""
        categories = list(metrics.keys())
        values = list(metrics.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Compliance Scores'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Compliance Metrics Radar Chart"
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig

    def plot_violation_timeline(
        self,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Plot violation timeline."""
        violations = []
        for metrics in self.metrics_history:
            for metric in ['bias', 'privacy', 'transparency', 'fairness']:
                score = getattr(metrics, f"{metric}_score")
                if score < 0.8:  # Threshold
                    violations.append({
                        'metric': metric,
                        'timestamp': metrics['timestamp'],
                        'score': score
                    })
        
        df = pd.DataFrame(violations)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig = px.scatter(
            df,
            x='timestamp',
            y='score',
            color='metric',
            title='Compliance Violations Timeline'
        )
        
        fig.add_hline(y=0.8, line_dash="dash", line_color="red")
        
        if save_path:
            fig.write_html(save_path)
        
        return fig 