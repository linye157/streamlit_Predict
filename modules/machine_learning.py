import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import io
import os
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
import time
import json
import plotly.express as px
import plotly.graph_objects as go

class MachineLearningModule:
    def __init__(self):
        # Initialize model dict
        if 'models' not in st.session_state:
            st.session_state.models = {}
        
        # Initialize model results
        if 'model_results' not in st.session_state:
            st.session_state.model_results = {}
    
    def prepare_data(self):
        """Prepare data for model training"""
        if st.session_state.train_data is None:
            st.error("请先加载训练数据")
            return None, None, None, None
        
        if 'num_targets' not in st.session_state:
            st.session_state.num_targets = 3  # Default to 3 targets
        
        # Get features and targets
        X = st.session_state.train_data.iloc[:, :-st.session_state.num_targets]
        y = st.session_state.train_data.iloc[:, -st.session_state.num_targets:]
        
        # Split data
        test_size = st.slider("测试集比例", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        random_state = st.number_input("随机种子", min_value=0, max_value=1000, value=42, key="prep_random_state")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        st.success(f"数据准备完成！训练集: {X_train.shape[0]} 行，测试集: {X_test.shape[0]} 行")
        
        return X_train, X_test, y_train, y_test
    
    def linear_regression(self, X_train, X_test, y_train, y_test, target_idx):
        """Train linear regression model"""
        st.subheader("线性回归 (LR) 模型")
        
        fit_intercept = st.checkbox("拟合截距", value=True)
        normalize = st.checkbox("归一化", value=False)
        
        if st.button("训练线性回归模型"):
            with st.spinner("正在训练线性回归模型..."):
                # Create and train model
                model = LinearRegression(fit_intercept=fit_intercept)
                
                # Start time
                start_time = time.time()
                
                # Train model
                model.fit(X_train, y_train.iloc[:, target_idx])
                
                # End time
                end_time = time.time()
                
                # Make predictions
                train_preds = model.predict(X_train)
                test_preds = model.predict(X_test)
                
                # Calculate metrics
                train_rmse = np.sqrt(mean_squared_error(y_train.iloc[:, target_idx], train_preds))
                test_rmse = np.sqrt(mean_squared_error(y_test.iloc[:, target_idx], test_preds))
                train_r2 = r2_score(y_train.iloc[:, target_idx], train_preds)
                test_r2 = r2_score(y_test.iloc[:, target_idx], test_preds)
                train_mae = mean_absolute_error(y_train.iloc[:, target_idx], train_preds)
                test_mae = mean_absolute_error(y_test.iloc[:, target_idx], test_preds)
                
                # Store model
                model_name = f"LR_Target_{target_idx}"
                st.session_state.models[model_name] = model
                
                # Store results
                st.session_state.model_results[model_name] = {
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'training_time': end_time - start_time,
                    'target_idx': target_idx,
                    'feature_importance': None,  # Not applicable for LR
                    'predictions': {
                        'train': train_preds.tolist(),
                        'test': test_preds.tolist(),
                        'train_actual': y_train.iloc[:, target_idx].values.tolist(),
                        'test_actual': y_test.iloc[:, target_idx].values.tolist()
                    }
                }
                
                # Display results
                self.display_model_results(model_name)
                
                # Save model
                self.save_model(model, model_name)
    
    def random_forest(self, X_train, X_test, y_train, y_test, target_idx):
        """Train random forest model"""
        st.subheader("随机森林 (RF) 模型")
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_estimators = st.slider("树的数量", min_value=10, max_value=300, value=100, step=10)
            max_depth = st.slider("最大深度", min_value=1, max_value=50, value=None)
            min_samples_split = st.slider("最小分裂样本数", min_value=2, max_value=20, value=2, step=1)
        
        with col2:
            min_samples_leaf = st.slider("最小叶子结点样本数", min_value=1, max_value=20, value=1, step=1)
            max_features = st.selectbox("最大特征数", ["sqrt", "log2", None], index=0)
            random_state = st.number_input("随机种子", min_value=0, max_value=1000, value=42, key="rf_random_state")
        
        if st.button("训练随机森林模型"):
            with st.spinner("正在训练随机森林模型..."):
                # Create and train model
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    random_state=random_state,
                    n_jobs=-1
                )
                
                # Start time
                start_time = time.time()
                
                # Train model
                model.fit(X_train, y_train.iloc[:, target_idx])
                
                # End time
                end_time = time.time()
                
                # Make predictions
                train_preds = model.predict(X_train)
                test_preds = model.predict(X_test)
                
                # Calculate metrics
                train_rmse = np.sqrt(mean_squared_error(y_train.iloc[:, target_idx], train_preds))
                test_rmse = np.sqrt(mean_squared_error(y_test.iloc[:, target_idx], test_preds))
                train_r2 = r2_score(y_train.iloc[:, target_idx], train_preds)
                test_r2 = r2_score(y_test.iloc[:, target_idx], test_preds)
                train_mae = mean_absolute_error(y_train.iloc[:, target_idx], train_preds)
                test_mae = mean_absolute_error(y_test.iloc[:, target_idx], test_preds)
                
                # Store model
                model_name = f"RF_Target_{target_idx}"
                st.session_state.models[model_name] = model
                
                # Store results
                st.session_state.model_results[model_name] = {
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'training_time': end_time - start_time,
                    'target_idx': target_idx,
                    'feature_importance': model.feature_importances_.tolist(),
                    'feature_names': X_train.columns.tolist(),
                    'predictions': {
                        'train': train_preds.tolist(),
                        'test': test_preds.tolist(),
                        'train_actual': y_train.iloc[:, target_idx].values.tolist(),
                        'test_actual': y_test.iloc[:, target_idx].values.tolist()
                    }
                }
                
                # Display results
                self.display_model_results(model_name)
                
                # Save model
                self.save_model(model, model_name)
    
    def gradient_boosting(self, X_train, X_test, y_train, y_test, target_idx):
        """Train gradient boosting model"""
        st.subheader("梯度提升 (GBR) 模型")
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_estimators = st.slider("树的数量 (GBR)", min_value=10, max_value=500, value=100, step=10)
            learning_rate = st.slider("学习率", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
            max_depth = st.slider("最大深度 (GBR)", min_value=1, max_value=20, value=3, step=1)
        
        with col2:
            min_samples_split = st.slider("最小分裂样本数 (GBR)", min_value=2, max_value=20, value=2, step=1)
            min_samples_leaf = st.slider("最小叶子结点样本数 (GBR)", min_value=1, max_value=20, value=1, step=1)
            random_state = st.number_input("随机种子 (GBR)", min_value=0, max_value=1000, value=42, key="gbr_random_state")
        
        if st.button("训练梯度提升模型"):
            with st.spinner("正在训练梯度提升模型..."):
                # Create and train model
                model = GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=random_state
                )
                
                # Start time
                start_time = time.time()
                
                # Train model
                model.fit(X_train, y_train.iloc[:, target_idx])
                
                # End time
                end_time = time.time()
                
                # Make predictions
                train_preds = model.predict(X_train)
                test_preds = model.predict(X_test)
                
                # Calculate metrics
                train_rmse = np.sqrt(mean_squared_error(y_train.iloc[:, target_idx], train_preds))
                test_rmse = np.sqrt(mean_squared_error(y_test.iloc[:, target_idx], test_preds))
                train_r2 = r2_score(y_train.iloc[:, target_idx], train_preds)
                test_r2 = r2_score(y_test.iloc[:, target_idx], test_preds)
                train_mae = mean_absolute_error(y_train.iloc[:, target_idx], train_preds)
                test_mae = mean_absolute_error(y_test.iloc[:, target_idx], test_preds)
                
                # Store model
                model_name = f"GBR_Target_{target_idx}"
                st.session_state.models[model_name] = model
                
                # Store results
                st.session_state.model_results[model_name] = {
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'training_time': end_time - start_time,
                    'target_idx': target_idx,
                    'feature_importance': model.feature_importances_.tolist(),
                    'feature_names': X_train.columns.tolist(),
                    'predictions': {
                        'train': train_preds.tolist(),
                        'test': test_preds.tolist(),
                        'train_actual': y_train.iloc[:, target_idx].values.tolist(),
                        'test_actual': y_test.iloc[:, target_idx].values.tolist()
                    }
                }
                
                # Display results
                self.display_model_results(model_name)
                
                # Save model
                self.save_model(model, model_name)
    
    def xgboost(self, X_train, X_test, y_train, y_test, target_idx):
        """Train XGBoost model"""
        st.subheader("XGBoost (XGBR) 模型")
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_estimators = st.slider("树的数量 (XGB)", min_value=10, max_value=500, value=100, step=10)
            learning_rate = st.slider("学习率 (XGB)", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
            max_depth = st.slider("最大深度 (XGB)", min_value=1, max_value=20, value=6, step=1)
        
        with col2:
            subsample = st.slider("子样本比例", min_value=0.1, max_value=1.0, value=1.0, step=0.1)
            colsample_bytree = st.slider("特征采样比例", min_value=0.1, max_value=1.0, value=1.0, step=0.1)
            random_state = st.number_input("随机种子 (XGB)", min_value=0, max_value=1000, value=42, key="xgb_random_state")
        
        if st.button("训练XGBoost模型"):
            with st.spinner("正在训练XGBoost模型..."):
                # Create and train model
                model = XGBRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    random_state=random_state,
                    n_jobs=-1
                )
                
                # Start time
                start_time = time.time()
                
                # Train model
                model.fit(X_train, y_train.iloc[:, target_idx])
                
                # End time
                end_time = time.time()
                
                # Make predictions
                train_preds = model.predict(X_train)
                test_preds = model.predict(X_test)
                
                # Calculate metrics
                train_rmse = np.sqrt(mean_squared_error(y_train.iloc[:, target_idx], train_preds))
                test_rmse = np.sqrt(mean_squared_error(y_test.iloc[:, target_idx], test_preds))
                train_r2 = r2_score(y_train.iloc[:, target_idx], train_preds)
                test_r2 = r2_score(y_test.iloc[:, target_idx], test_preds)
                train_mae = mean_absolute_error(y_train.iloc[:, target_idx], train_preds)
                test_mae = mean_absolute_error(y_test.iloc[:, target_idx], test_preds)
                
                # Store model
                model_name = f"XGB_Target_{target_idx}"
                st.session_state.models[model_name] = model
                
                # Store results
                st.session_state.model_results[model_name] = {
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'training_time': end_time - start_time,
                    'target_idx': target_idx,
                    'feature_importance': model.feature_importances_.tolist(),
                    'feature_names': X_train.columns.tolist(),
                    'predictions': {
                        'train': train_preds.tolist(),
                        'test': test_preds.tolist(),
                        'train_actual': y_train.iloc[:, target_idx].values.tolist(),
                        'test_actual': y_test.iloc[:, target_idx].values.tolist()
                    }
                }
                
                # Display results
                self.display_model_results(model_name)
                
                # Save model
                self.save_model(model, model_name)
    
    def support_vector_regression(self, X_train, X_test, y_train, y_test, target_idx):
        """Train Support Vector Regression model"""
        st.subheader("支持向量回归 (SVR) 模型")
        
        col1, col2 = st.columns(2)
        
        with col1:
            kernel = st.selectbox("核函数", ["linear", "poly", "rbf", "sigmoid"], index=2)
            C = st.number_input("正则化参数 C", min_value=0.01, max_value=100.0, value=1.0, step=0.1)
        
        with col2:
            epsilon = st.number_input("epsilon", min_value=0.01, max_value=1.0, value=0.1, step=0.01)
            gamma = st.selectbox("gamma", ["scale", "auto"], index=0)
        
        if st.button("训练SVR模型"):
            with st.spinner("正在训练SVR模型..."):
                # Create and train model
                model = SVR(
                    kernel=kernel,
                    C=C,
                    epsilon=epsilon,
                    gamma=gamma
                )
                
                # Start time
                start_time = time.time()
                
                # Train model
                model.fit(X_train, y_train.iloc[:, target_idx])
                
                # End time
                end_time = time.time()
                
                # Make predictions
                train_preds = model.predict(X_train)
                test_preds = model.predict(X_test)
                
                # Calculate metrics
                train_rmse = np.sqrt(mean_squared_error(y_train.iloc[:, target_idx], train_preds))
                test_rmse = np.sqrt(mean_squared_error(y_test.iloc[:, target_idx], test_preds))
                train_r2 = r2_score(y_train.iloc[:, target_idx], train_preds)
                test_r2 = r2_score(y_test.iloc[:, target_idx], test_preds)
                train_mae = mean_absolute_error(y_train.iloc[:, target_idx], train_preds)
                test_mae = mean_absolute_error(y_test.iloc[:, target_idx], test_preds)
                
                # Store model
                model_name = f"SVR_Target_{target_idx}"
                st.session_state.models[model_name] = model
                
                # Store results
                st.session_state.model_results[model_name] = {
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'training_time': end_time - start_time,
                    'target_idx': target_idx,
                    'feature_importance': None,  # Not applicable for SVR
                    'predictions': {
                        'train': train_preds.tolist(),
                        'test': test_preds.tolist(),
                        'train_actual': y_train.iloc[:, target_idx].values.tolist(),
                        'test_actual': y_test.iloc[:, target_idx].values.tolist()
                    }
                }
                
                # Display results
                self.display_model_results(model_name)
                
                # Save model
                self.save_model(model, model_name)
    
    def neural_network(self, X_train, X_test, y_train, y_test, target_idx):
        """Train Neural Network model"""
        st.subheader("人工神经网络 (ANN) 模型")
        
        col1, col2 = st.columns(2)
        
        with col1:
            hidden_layer_sizes = st.text_input("隐藏层大小", "(100,)")
            activation = st.selectbox("激活函数", ["identity", "logistic", "tanh", "relu"], index=3)
            solver = st.selectbox("优化器", ["adam", "sgd", "lbfgs"], index=0)
        
        with col2:
            alpha = st.number_input("正则化参数 alpha", min_value=0.0001, max_value=1.0, value=0.0001, step=0.0001, format="%.4f")
            max_iter = st.number_input("最大迭代次数", min_value=100, max_value=10000, value=1000, step=100)
            random_state = st.number_input("随机种子 (ANN)", min_value=0, max_value=1000, value=42, key="ann_random_state")
        
        if st.button("训练神经网络模型"):
            with st.spinner("正在训练神经网络模型..."):
                try:
                    # Parse hidden_layer_sizes
                    hidden_layer_sizes = eval(hidden_layer_sizes)
                    
                    # Create and train model
                    model = MLPRegressor(
                        hidden_layer_sizes=hidden_layer_sizes,
                        activation=activation,
                        solver=solver,
                        alpha=alpha,
                        max_iter=max_iter,
                        random_state=random_state
                    )
                    
                    # Start time
                    start_time = time.time()
                    
                    # Train model
                    model.fit(X_train, y_train.iloc[:, target_idx])
                    
                    # End time
                    end_time = time.time()
                    
                    # Make predictions
                    train_preds = model.predict(X_train)
                    test_preds = model.predict(X_test)
                    
                    # Calculate metrics
                    train_rmse = np.sqrt(mean_squared_error(y_train.iloc[:, target_idx], train_preds))
                    test_rmse = np.sqrt(mean_squared_error(y_test.iloc[:, target_idx], test_preds))
                    train_r2 = r2_score(y_train.iloc[:, target_idx], train_preds)
                    test_r2 = r2_score(y_test.iloc[:, target_idx], test_preds)
                    train_mae = mean_absolute_error(y_train.iloc[:, target_idx], train_preds)
                    test_mae = mean_absolute_error(y_test.iloc[:, target_idx], test_preds)
                    
                    # Store model
                    model_name = f"ANN_Target_{target_idx}"
                    st.session_state.models[model_name] = model
                    
                    # Store results
                    st.session_state.model_results[model_name] = {
                        'train_rmse': train_rmse,
                        'test_rmse': test_rmse,
                        'train_r2': train_r2,
                        'test_r2': test_r2,
                        'train_mae': train_mae,
                        'test_mae': test_mae,
                        'training_time': end_time - start_time,
                        'target_idx': target_idx,
                        'feature_importance': None,  # Not applicable for ANN
                        'predictions': {
                            'train': train_preds.tolist(),
                            'test': test_preds.tolist(),
                            'train_actual': y_train.iloc[:, target_idx].values.tolist(),
                            'test_actual': y_test.iloc[:, target_idx].values.tolist()
                        }
                    }
                    
                    # Display results
                    self.display_model_results(model_name)
                    
                    # Save model
                    self.save_model(model, model_name)
                    
                except Exception as e:
                    st.error(f"训练神经网络时出错: {str(e)}")
    
    def display_model_results(self, model_name):
        """Display model results"""
        if model_name in st.session_state.model_results:
            results = st.session_state.model_results[model_name]
            
            st.subheader(f"模型结果: {model_name}")
            
            # Create metrics display
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("训练 RMSE", f"{results['train_rmse']:.4f}")
                st.metric("测试 RMSE", f"{results['test_rmse']:.4f}")
            
            with col2:
                st.metric("训练 R²", f"{results['train_r2']:.4f}")
                st.metric("测试 R²", f"{results['test_r2']:.4f}")
            
            with col3:
                st.metric("训练 MAE", f"{results['train_mae']:.4f}")
                st.metric("测试 MAE", f"{results['test_mae']:.4f}")
            
            with col4:
                st.metric("训练时间", f"{results['training_time']:.2f} 秒")
            
            # Visualization of predictions vs actual
            st.subheader("预测结果可视化")
            
            # Create scatter plot
            fig = go.Figure()
            
            # Add train data
            fig.add_trace(go.Scatter(
                x=results['predictions']['train_actual'],
                y=results['predictions']['train'],
                mode='markers',
                name='训练集',
                marker=dict(color='blue', size=8, opacity=0.6)
            ))
            
            # Add test data
            fig.add_trace(go.Scatter(
                x=results['predictions']['test_actual'],
                y=results['predictions']['test'],
                mode='markers',
                name='测试集',
                marker=dict(color='red', size=8, opacity=0.6)
            ))
            
            # Add perfect prediction line
            min_val = min(min(results['predictions']['train_actual']), min(results['predictions']['test_actual']))
            max_val = max(max(results['predictions']['train_actual']), max(results['predictions']['test_actual']))
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
            
            # Feature importance visualization if available
            if results['feature_importance'] is not None:
                st.subheader("特征重要性")
                
                # Create feature importance dataframe
                feature_imp_df = pd.DataFrame({
                    'Feature': results['feature_names'],
                    'Importance': results['feature_importance']
                })
                
                # Sort by importance
                feature_imp_df = feature_imp_df.sort_values('Importance', ascending=False)
                
                # Display top 15 features
                top_n = min(15, len(feature_imp_df))
                top_features = feature_imp_df.head(top_n)
                
                # Create bar chart
                fig = px.bar(
                    top_features,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title=f'前 {top_n} 特征重要性'
                )
                
                fig.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def save_model(self, model, model_name):
        """Save model to file"""
        try:
            # Create models directory if it doesn't exist
            os.makedirs('models', exist_ok=True)
            
            # Save model
            model_path = os.path.join('models', f"{model_name}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            st.success(f"模型已保存至 {model_path}")
        except Exception as e:
            st.error(f"保存模型时出错: {str(e)}")
    
    def load_model(self, model_path):
        """Load model from file"""
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            st.error(f"加载模型时出错: {str(e)}")
            return None
    
    def model_comparison(self):
        """Compare trained models"""
        if len(st.session_state.model_results) == 0:
            st.info("请先训练模型")
            return
        
        st.subheader("模型比较")
        
        # Group models by target
        targets = {}
        for model_name, results in st.session_state.model_results.items():
            target_idx = results['target_idx']
            if target_idx not in targets:
                targets[target_idx] = []
            targets[target_idx].append(model_name)
        
        # Select target to compare
        if len(targets) > 0:
            target_choice = st.selectbox(
                "选择目标变量",
                list(targets.keys()),
                format_func=lambda x: f"目标 {x}"
            )
            
            if target_choice is not None:
                # Get models for selected target
                target_models = targets[target_choice]
                
                # Create comparison dataframe
                comparison_data = []
                for model_name in target_models:
                    results = st.session_state.model_results[model_name]
                    comparison_data.append({
                        '模型': model_name,
                        '训练 RMSE': results['train_rmse'],
                        '测试 RMSE': results['test_rmse'],
                        '训练 R²': results['train_r2'],
                        '测试 R²': results['test_r2'],
                        '训练 MAE': results['train_mae'],
                        '测试 MAE': results['test_mae'],
                        '训练时间': results['training_time']
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df.set_index('模型'))
                
                # Create bar chart for test metrics
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
    
    def render(self):
        st.title("机器学习")
        
        if st.session_state.train_data is None:
            st.warning("请先加载训练数据")
            return
        
        tab1, tab2 = st.tabs(["模型训练", "模型比较"])
        
        with tab1:
            # Prepare data
            st.subheader("准备数据")
            X_train, X_test, y_train, y_test = self.prepare_data()
            
            if X_train is not None:
                # Select target
                available_targets = y_train.columns.tolist()
                target_idx = st.selectbox(
                    "选择目标变量",
                    range(len(available_targets)),
                    format_func=lambda i: available_targets[i]
                )
                
                # Select model
                model_choice = st.selectbox(
                    "选择模型",
                    ["线性回归 (LR)", "随机森林 (RF)", "梯度提升 (GBR)", "XGBoost (XGBR)", "支持向量机 (SVR)", "人工神经网络 (ANN)"],
                    index=0
                )
                
                # Train selected model
                if model_choice == "线性回归 (LR)":
                    self.linear_regression(X_train, X_test, y_train, y_test, target_idx)
                elif model_choice == "随机森林 (RF)":
                    self.random_forest(X_train, X_test, y_train, y_test, target_idx)
                elif model_choice == "梯度提升 (GBR)":
                    self.gradient_boosting(X_train, X_test, y_train, y_test, target_idx)
                elif model_choice == "XGBoost (XGBR)":
                    self.xgboost(X_train, X_test, y_train, y_test, target_idx)
                elif model_choice == "支持向量机 (SVR)":
                    self.support_vector_regression(X_train, X_test, y_train, y_test, target_idx)
                elif model_choice == "人工神经网络 (ANN)":
                    self.neural_network(X_train, X_test, y_train, y_test, target_idx)
        
        with tab2:
            self.model_comparison() 