import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
from datetime import datetime
import os
import json

class ReportModule:
    def __init__(self):
        pass
    
    def create_report(self):
        """Create custom report"""
        st.subheader("报表订制")
        
        # Check if there's data available
        if st.session_state.train_data is None:
            st.warning("请先加载数据")
            return
        
        # Report title
        report_title = st.text_input("报表标题", "机器学习模型评估报告")
        
        # Report sections
        sections = st.multiselect(
            "选择报表内容",
            ["数据摘要", "模型性能", "模型对比", "预测结果", "特征重要性"],
            default=["数据摘要", "模型性能", "模型对比"]
        )
        
        # Select models for report
        if "模型性能" in sections or "模型对比" in sections or "预测结果" in sections or "特征重要性" in sections:
            if 'model_results' in st.session_state and st.session_state.model_results:
                available_models = list(st.session_state.model_results.keys())
                selected_models = st.multiselect(
                    "选择要包含在报表中的模型",
                    available_models,
                    default=available_models[:3] if len(available_models) > 3 else available_models
                )
            else:
                st.warning("尚无训练模型可用于报表")
                selected_models = []
        
        # Generate report button
        if st.button("生成报表"):
            if not sections:
                st.warning("请至少选择一个报表内容区域")
                return
            
            # Create report container
            report_container = st.container()
            
            with report_container:
                st.title(report_title)
                st.markdown(f"*生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
                st.markdown("---")
                
                # Data summary section
                if "数据摘要" in sections:
                    st.header("1. 数据摘要")
                    
                    # Show data shapes
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.session_state.train_data is not None:
                            st.subheader("训练数据")
                            st.write(f"形状: {st.session_state.train_data.shape[0]} 行, {st.session_state.train_data.shape[1]} 列")
                            st.dataframe(st.session_state.train_data.describe())
                    
                    with col2:
                        if st.session_state.test_data is not None:
                            st.subheader("测试数据")
                            st.write(f"形状: {st.session_state.test_data.shape[0]} 行, {st.session_state.test_data.shape[1]} 列")
                            st.dataframe(st.session_state.test_data.describe())
                    
                    # Missing values
                    st.subheader("缺失值统计")
                    if st.session_state.train_data is not None:
                        missing = st.session_state.train_data.isnull().sum()
                        missing = missing[missing > 0]
                        if len(missing) > 0:
                            missing_df = pd.DataFrame({
                                '缺失值数量': missing,
                                '缺失比例': missing / len(st.session_state.train_data) * 100
                            })
                            st.dataframe(missing_df)
                        else:
                            st.write("训练数据无缺失值")
                
                # Model performance section
                if "模型性能" in sections and selected_models:
                    st.header("2. 模型性能")
                    
                    for model_name in selected_models:
                        if model_name in st.session_state.model_results:
                            results = st.session_state.model_results[model_name]
                            
                            st.subheader(f"模型: {model_name}")
                            
                            # Create metrics in two columns
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("训练 RMSE", f"{results['train_rmse']:.4f}")
                                st.metric("训练 R²", f"{results['train_r2']:.4f}")
                                st.metric("训练 MAE", f"{results['train_mae']:.4f}")
                            
                            with col2:
                                st.metric("测试 RMSE", f"{results['test_rmse']:.4f}")
                                st.metric("测试 R²", f"{results['test_r2']:.4f}")
                                st.metric("测试 MAE", f"{results['test_mae']:.4f}")
                            
                            # Create scatter plot
                            if 'predictions' in results:
                                pred_data = results['predictions']
                                
                                fig = go.Figure()
                                
                                # Add train data
                                fig.add_trace(go.Scatter(
                                    x=pred_data['train_actual'],
                                    y=pred_data['train'],
                                    mode='markers',
                                    name='训练集',
                                    marker=dict(color='blue', size=8, opacity=0.6)
                                ))
                                
                                # Add test data
                                fig.add_trace(go.Scatter(
                                    x=pred_data['test_actual'],
                                    y=pred_data['test'],
                                    mode='markers',
                                    name='测试集',
                                    marker=dict(color='red', size=8, opacity=0.6)
                                ))
                                
                                # Add perfect prediction line
                                min_val = min(min(pred_data['train_actual']), min(pred_data['test_actual']))
                                max_val = max(max(pred_data['train_actual']), max(pred_data['test_actual']))
                                fig.add_trace(go.Scatter(
                                    x=[min_val, max_val],
                                    y=[min_val, max_val],
                                    mode='lines',
                                    name='完美预测',
                                    line=dict(color='black', dash='dash')
                                ))
                                
                                fig.update_layout(
                                    title='实际值 vs 预测值',
                                    xaxis_title='实际值',
                                    yaxis_title='预测值',
                                    legend_title='数据集',
                                    template='plotly_white'
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                
                # Model comparison section
                if "模型对比" in sections and len(selected_models) > 1:
                    st.header("3. 模型对比")
                    
                    # Create comparison dataframe
                    comparison_data = []
                    for model_name in selected_models:
                        if model_name in st.session_state.model_results:
                            results = st.session_state.model_results[model_name]
                            comparison_data.append({
                                '模型': model_name,
                                '训练 RMSE': results['train_rmse'],
                                '测试 RMSE': results['test_rmse'],
                                '训练 R²': results['train_r2'],
                                '测试 R²': results['test_r2'],
                                '训练 MAE': results['train_mae'],
                                '测试 MAE': results['test_mae'],
                                '训练时间': results.get('training_time', 0)
                            })
                    
                    if comparison_data:
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df.set_index('模型'))
                        
                        # Create bar charts for test metrics
                        metrics = ['测试 RMSE', '测试 MAE', '测试 R²']
                        for metric in metrics:
                            # Sort by metric (ascending for error metrics, descending for R²)
                            ascending = metric != '测试 R²'
                            sorted_df = comparison_df.sort_values(metric, ascending=ascending)
                            
                            fig = px.bar(
                                sorted_df,
                                x='模型',
                                y=metric,
                                title=f'模型比较 - {metric}',
                                color='模型'
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                
                # Prediction results section
                if "预测结果" in sections and selected_models:
                    st.header("4. 预测结果")
                    
                    # Create tabs for each model
                    if len(selected_models) > 0:
                        tabs = st.tabs([f"模型: {name}" for name in selected_models])
                        
                        for i, model_name in enumerate(selected_models):
                            if model_name in st.session_state.model_results:
                                results = st.session_state.model_results[model_name]
                                
                                with tabs[i]:
                                    if 'predictions' in results:
                                        pred_data = results['predictions']
                                        
                                        # Create sample prediction results table
                                        test_df = pd.DataFrame({
                                            '实际值': pred_data['test_actual'][:10],
                                            '预测值': pred_data['test'][:10],
                                            '误差': np.array(pred_data['test'][:10]) - np.array(pred_data['test_actual'][:10]),
                                            '绝对误差': np.abs(np.array(pred_data['test'][:10]) - np.array(pred_data['test_actual'][:10]))
                                        })
                                        
                                        st.subheader("测试集预测样本 (前10条)")
                                        st.dataframe(test_df)
                                        
                                        # Error histogram
                                        st.subheader("误差分布")
                                        
                                        error = np.array(pred_data['test']) - np.array(pred_data['test_actual'])
                                        
                                        fig = px.histogram(
                                            x=error, 
                                            nbins=30,
                                            labels={'x': '误差', 'y': '频数'},
                                            title='预测误差分布',
                                            color_discrete_sequence=['indianred']
                                        )
                                        
                                        fig.add_vline(x=0, line_dash="dash", line_color="black")
                                        
                                        st.plotly_chart(fig, use_container_width=True)
                
                # Feature importance section
                if "特征重要性" in sections and selected_models:
                    st.header("5. 特征重要性")
                    
                    for model_name in selected_models:
                        if model_name in st.session_state.model_results:
                            results = st.session_state.model_results[model_name]
                            
                            # Check if feature importance exists
                            if 'feature_importance' in results and results['feature_importance'] is not None:
                                st.subheader(f"模型: {model_name}")
                                
                                # Create dataframe for plotting
                                feature_names = results.get('feature_names', [f"Feature {i}" for i in range(len(results['feature_importance']))])
                                importance_df = pd.DataFrame({
                                    'Feature': feature_names,
                                    'Importance': results['feature_importance']
                                })
                                
                                # Sort by importance
                                importance_df = importance_df.sort_values('Importance', ascending=False)
                                
                                # Display top 15 features
                                top_n = min(15, len(importance_df))
                                top_features = importance_df.head(top_n)
                                
                                # Create bar chart
                                fig = px.bar(
                                    top_features,
                                    x='Importance',
                                    y='Feature',
                                    orientation='h',
                                    title=f"特征重要性 (前 {top_n} 项)"
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                st.markdown("*报表结束*")
            
            # Store report in session state
            if 'reports' not in st.session_state:
                st.session_state.reports = {}
            
            report_id = f"report_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            st.session_state.reports[report_id] = {
                'title': report_title,
                'sections': sections,
                'models': selected_models,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            st.success("报表生成成功！")
    
    def generate_report(self, report_format="html"):
        """Generate report in various formats"""
        st.subheader("报表导出")
        
        # Check if there are available reports
        if 'reports' not in st.session_state or not st.session_state.reports:
            st.warning("尚无可用报表，请先创建报表")
            return
        
        # Select report to export
        report_ids = list(st.session_state.reports.keys())
        report_options = [f"{st.session_state.reports[rid]['title']} ({st.session_state.reports[rid]['timestamp']})" for rid in report_ids]
        
        selected_report_idx = st.selectbox("选择要导出的报表", range(len(report_options)), format_func=lambda i: report_options[i])
        selected_report_id = report_ids[selected_report_idx]
        
        # Select export format
        export_format = st.selectbox(
            "导出格式",
            ["HTML", "PDF", "Excel", "Word"],
            index=0
        )
        
        if st.button("导出报表"):
            st.info("正在准备下载...")
            
            # Create download function based on format
            if export_format == "HTML":
                html_report = self.create_html_report(selected_report_id)
                
                # Create download button
                st.download_button(
                    label="下载HTML报表",
                    data=html_report,
                    file_name=f"{st.session_state.reports[selected_report_id]['title']}.html",
                    mime="text/html"
                )
            
            elif export_format == "Excel":
                excel_report = self.create_excel_report(selected_report_id)
                
                # Create download button
                st.download_button(
                    label="下载Excel报表",
                    data=excel_report,
                    file_name=f"{st.session_state.reports[selected_report_id]['title']}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            elif export_format == "PDF":
                st.warning("PDF导出功能正在开发中")
            
            elif export_format == "Word":
                st.warning("Word导出功能正在开发中")
    
    def create_html_report(self, report_id):
        """Create HTML report"""
        import plotly.io as pio
        from jinja2 import Template
        
        report_info = st.session_state.reports[report_id]
        
        # HTML template
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{ title }}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }
                h1, h2, h3 {
                    color: #2C3E50;
                }
                h1 {
                    border-bottom: 2px solid #3498DB;
                    padding-bottom: 10px;
                }
                h2 {
                    margin-top: 30px;
                    border-bottom: 1px solid #BDC3C7;
                    padding-bottom: 5px;
                }
                table {
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }
                th, td {
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #E1E1E1;
                }
                th {
                    background-color: #F5F5F5;
                }
                .metric {
                    display: inline-block;
                    margin: 10px;
                    padding: 15px;
                    background-color: #F8F9F9;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }
                .metric h3 {
                    margin: 0;
                    font-size: 14px;
                    color: #7F8C8D;
                }
                .metric p {
                    margin: 5px 0 0 0;
                    font-size: 24px;
                    font-weight: bold;
                }
                .plot {
                    margin: 30px 0;
                }
                .footer {
                    margin-top: 50px;
                    padding-top: 20px;
                    border-top: 1px solid #BDC3C7;
                    text-align: center;
                    font-size: 14px;
                    color: #7F8C8D;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{{ title }}</h1>
                <p><em>生成时间: {{ timestamp }}</em></p>
                
                {{ report_content }}
                
                <div class="footer">
                    <p>报表生成于 {{ timestamp }}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Build report content
        report_content = []
        sections = report_info['sections']
        selected_models = report_info['models']
        
        # Data summary section
        if "数据摘要" in sections:
            report_content.append("<h2>1. 数据摘要</h2>")
            
            if st.session_state.train_data is not None:
                report_content.append("<h3>训练数据</h3>")
                report_content.append(f"<p>形状: {st.session_state.train_data.shape[0]} 行, {st.session_state.train_data.shape[1]} 列</p>")
                report_content.append(st.session_state.train_data.describe().to_html())
            
            if st.session_state.test_data is not None:
                report_content.append("<h3>测试数据</h3>")
                report_content.append(f"<p>形状: {st.session_state.test_data.shape[0]} 行, {st.session_state.test_data.shape[1]} 列</p>")
                report_content.append(st.session_state.test_data.describe().to_html())
            
            # Missing values
            report_content.append("<h3>缺失值统计</h3>")
            if st.session_state.train_data is not None:
                missing = st.session_state.train_data.isnull().sum()
                missing = missing[missing > 0]
                if len(missing) > 0:
                    missing_df = pd.DataFrame({
                        '缺失值数量': missing,
                        '缺失比例': missing / len(st.session_state.train_data) * 100
                    })
                    report_content.append(missing_df.to_html())
                else:
                    report_content.append("<p>训练数据无缺失值</p>")
        
        # Model performance section
        if "模型性能" in sections and selected_models:
            report_content.append("<h2>2. 模型性能</h2>")
            
            for model_name in selected_models:
                if model_name in st.session_state.model_results:
                    results = st.session_state.model_results[model_name]
                    
                    report_content.append(f"<h3>模型: {model_name}</h3>")
                    
                    # Create metrics
                    metrics_html = """
                    <div style="display: flex; flex-wrap: wrap;">
                        <div class="metric">
                            <h3>训练 RMSE</h3>
                            <p>{:.4f}</p>
                        </div>
                        <div class="metric">
                            <h3>训练 R²</h3>
                            <p>{:.4f}</p>
                        </div>
                        <div class="metric">
                            <h3>训练 MAE</h3>
                            <p>{:.4f}</p>
                        </div>
                        <div class="metric">
                            <h3>测试 RMSE</h3>
                            <p>{:.4f}</p>
                        </div>
                        <div class="metric">
                            <h3>测试 R²</h3>
                            <p>{:.4f}</p>
                        </div>
                        <div class="metric">
                            <h3>测试 MAE</h3>
                            <p>{:.4f}</p>
                        </div>
                    </div>
                    """.format(
                        results['train_rmse'], results['train_r2'], results['train_mae'],
                        results['test_rmse'], results['test_r2'], results['test_mae']
                    )
                    
                    report_content.append(metrics_html)
                    
                    # Create scatter plot
                    if 'predictions' in results:
                        pred_data = results['predictions']
                        
                        fig = go.Figure()
                        
                        # Add train data
                        fig.add_trace(go.Scatter(
                            x=pred_data['train_actual'],
                            y=pred_data['train'],
                            mode='markers',
                            name='训练集',
                            marker=dict(color='blue', size=8, opacity=0.6)
                        ))
                        
                        # Add test data
                        fig.add_trace(go.Scatter(
                            x=pred_data['test_actual'],
                            y=pred_data['test'],
                            mode='markers',
                            name='测试集',
                            marker=dict(color='red', size=8, opacity=0.6)
                        ))
                        
                        # Add perfect prediction line
                        min_val = min(min(pred_data['train_actual']), min(pred_data['test_actual']))
                        max_val = max(max(pred_data['train_actual']), max(pred_data['test_actual']))
                        fig.add_trace(go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode='lines',
                            name='完美预测',
                            line=dict(color='black', dash='dash')
                        ))
                        
                        fig.update_layout(
                            title='实际值 vs 预测值',
                            xaxis_title='实际值',
                            yaxis_title='预测值',
                            legend_title='数据集',
                            template='plotly_white',
                            height=500
                        )
                        
                        plot_div = f"""<div class="plot">{pio.to_html(fig, include_plotlyjs=False, full_html=False)}</div>"""
                        report_content.append(plot_div)
        
        # Combine all sections
        html_content = Template(html_template).render(
            title=report_info['title'],
            timestamp=report_info['timestamp'],
            report_content='\n'.join(report_content)
        )
        
        return html_content
    
    def create_excel_report(self, report_id):
        """Create Excel report"""
        import pandas as pd
        from io import BytesIO
        
        report_info = st.session_state.reports[report_id]
        selected_models = report_info['models']
        
        # Create Excel writer
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Write report info
            info_df = pd.DataFrame({
                '报表标题': [report_info['title']],
                '生成时间': [report_info['timestamp']],
                '包含章节': [', '.join(report_info['sections'])],
                '包含模型': [', '.join(report_info['models'])]
            })
            info_df.to_excel(writer, sheet_name='报表信息', index=False)
            
            # Write data summary
            if "数据摘要" in report_info['sections']:
                if st.session_state.train_data is not None:
                    st.session_state.train_data.describe().to_excel(writer, sheet_name='训练数据统计')
                if st.session_state.test_data is not None:
                    st.session_state.test_data.describe().to_excel(writer, sheet_name='测试数据统计')
            
            # Write model metrics
            if "模型性能" in report_info['sections'] or "模型对比" in report_info['sections']:
                # Create comparison dataframe
                comparison_data = []
                for model_name in selected_models:
                    if model_name in st.session_state.model_results:
                        results = st.session_state.model_results[model_name]
                        comparison_data.append({
                            '模型': model_name,
                            '训练 RMSE': results['train_rmse'],
                            '测试 RMSE': results['test_rmse'],
                            '训练 R²': results['train_r2'],
                            '测试 R²': results['test_r2'],
                            '训练 MAE': results['train_mae'],
                            '测试 MAE': results['test_mae'],
                            '训练时间': results.get('training_time', 0)
                        })
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    comparison_df.to_excel(writer, sheet_name='模型比较', index=False)
            
            # Write prediction results
            if "预测结果" in report_info['sections']:
                for model_name in selected_models:
                    if model_name in st.session_state.model_results:
                        results = st.session_state.model_results[model_name]
                        if 'predictions' in results:
                            pred_data = results['predictions']
                            
                            # Create predictions dataframe
                            pred_df = pd.DataFrame({
                                '实际值 (训练)': pred_data['train_actual'],
                                '预测值 (训练)': pred_data['train'],
                                '误差 (训练)': np.array(pred_data['train']) - np.array(pred_data['train_actual']),
                                '实际值 (测试)': pred_data['test_actual'],
                                '预测值 (测试)': pred_data['test'],
                                '误差 (测试)': np.array(pred_data['test']) - np.array(pred_data['test_actual'])
                            })
                            
                            # Limit to first 1000 rows to avoid Excel limitations
                            if len(pred_df) > 1000:
                                pred_df = pred_df.iloc[:1000]
                            
                            # Sheet name must be <= 31 characters
                            sheet_name = f"{model_name}_预测"
                            if len(sheet_name) > 31:
                                sheet_name = sheet_name[:31]
                            
                            pred_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Write feature importance
            if "特征重要性" in report_info['sections']:
                for model_name in selected_models:
                    if model_name in st.session_state.model_results:
                        results = st.session_state.model_results[model_name]
                        
                        # Check if feature importance exists
                        if 'feature_importance' in results and results['feature_importance'] is not None:
                            # Create dataframe for feature importance
                            feature_names = results.get('feature_names', [f"Feature {i}" for i in range(len(results['feature_importance']))])
                            importance_df = pd.DataFrame({
                                'Feature': feature_names,
                                'Importance': results['feature_importance']
                            })
                            
                            # Sort by importance
                            importance_df = importance_df.sort_values('Importance', ascending=False)
                            
                            # Sheet name must be <= 31 characters
                            sheet_name = f"{model_name}_特征重要性"
                            if len(sheet_name) > 31:
                                sheet_name = sheet_name[:31]
                            
                            importance_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        # Get the Excel data
        excel_data = output.getvalue()
        
        return excel_data
    
    def report_storage(self):
        """Store and retrieve reports"""
        st.subheader("报表存储")
        
        # Check if there are available reports
        if 'reports' not in st.session_state or not st.session_state.reports:
            st.warning("尚无可用报表，请先创建报表")
            return
        
        # Display saved reports
        st.write("已保存的报表:")
        
        # Create table of reports
        report_data = []
        for report_id, report_info in st.session_state.reports.items():
            report_data.append({
                'ID': report_id,
                '标题': report_info['title'],
                '生成时间': report_info['timestamp'],
                '包含章节': len(report_info['sections']),
                '包含模型': len(report_info['models'])
            })
        
        report_df = pd.DataFrame(report_data)
        st.dataframe(report_df)
        
        # Delete report option
        if len(report_data) > 0:
            report_to_delete = st.selectbox(
                "选择要删除的报表",
                report_data,
                format_func=lambda x: f"{x['标题']} ({x['生成时间']})"
            )
            
            if st.button("删除报表"):
                report_id = report_to_delete['ID']
                if report_id in st.session_state.reports:
                    del st.session_state.reports[report_id]
                    st.success(f"报表 '{report_to_delete['标题']}' 已删除")
                    st.rerun()
    
    def report_comparison(self):
        """Compare reports from different time periods"""
        st.subheader("报表对比")
        
        # Check if there are available reports
        if 'reports' not in st.session_state or len(st.session_state.reports) < 2:
            st.warning("至少需要两个报表才能进行对比")
            return
        
        # Select reports to compare
        report_options = [(rid, f"{info['title']} ({info['timestamp']})") 
                         for rid, info in st.session_state.reports.items()]
        
        col1, col2 = st.columns(2)
        
        with col1:
            report1_idx = st.selectbox(
                "报表 1",
                range(len(report_options)),
                format_func=lambda i: report_options[i][1]
            )
            report1_id = report_options[report1_idx][0]
        
        with col2:
            report2_idx = st.selectbox(
                "报表 2",
                range(len(report_options)),
                format_func=lambda i: report_options[i][1],
                index=min(1, len(report_options)-1)
            )
            report2_id = report_options[report2_idx][0]
        
        if report1_id == report2_id:
            st.warning("请选择两个不同的报表进行对比")
            return
        
        # Get report information
        report1_info = st.session_state.reports[report1_id]
        report2_info = st.session_state.reports[report2_id]
        
        # Compare reports button
        if st.button("对比报表"):
            st.subheader("报表对比结果")
            
            # Compare basic information
            st.write("### 基本信息对比")
            info_comparison = pd.DataFrame({
                '指标': ['标题', '生成时间', '包含章节数', '包含模型数'],
                '报表 1': [
                    report1_info['title'],
                    report1_info['timestamp'],
                    len(report1_info['sections']),
                    len(report1_info['models'])
                ],
                '报表 2': [
                    report2_info['title'],
                    report2_info['timestamp'],
                    len(report2_info['sections']),
                    len(report2_info['models'])
                ]
            })
            
            st.dataframe(info_comparison.set_index('指标'))
            
            # Compare model metrics if both reports have model comparison section
            if ("模型性能" in report1_info['sections'] or "模型对比" in report1_info['sections']) and \
               ("模型性能" in report2_info['sections'] or "模型对比" in report2_info['sections']):
                
                st.write("### 模型性能对比")
                
                # Find common models between reports
                common_models = set(report1_info['models']).intersection(set(report2_info['models']))
                
                if common_models:
                    for model_name in common_models:
                        if model_name in st.session_state.model_results:
                            st.write(f"#### 模型: {model_name}")
                            
                            # Get model metrics for each report
                            model_results = st.session_state.model_results[model_name]
                            
                            metrics_comparison = pd.DataFrame({
                                '指标': ['测试 RMSE', '测试 R²', '测试 MAE'],
                                '值': [
                                    model_results['test_rmse'],
                                    model_results['test_r2'],
                                    model_results['test_mae']
                                ]
                            })
                            
                            st.dataframe(metrics_comparison.set_index('指标'))
                else:
                    st.warning("两个报表没有共同的模型可供对比")
    
    def render(self):
        st.title("报表")
        
        tab1, tab2, tab3, tab4 = st.tabs(["报表订制", "报表下载", "报表存储", "报表对比"])
        
        with tab1:
            self.create_report()
        
        with tab2:
            self.generate_report()
        
        with tab3:
            self.report_storage()
        
        with tab4:
            self.report_comparison() 