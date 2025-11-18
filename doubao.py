"""
使用豆包大模型API判断中文语义完整性
模型: doubao-1-5-pro-32k-250115
"""

import requests
import json
from typing import Dict, Tuple


class DoubaoCompletenessChecker:
    """豆包模型语义完整性检测器"""
    
    def __init__(self, api_key: str, endpoint: str ):
        """
        初始化检测器
        
        Args:
            api_key: 火山引擎API密钥
            endpoint: API端点
        """
        self.api_key = api_key
        # 火山引擎豆包API endpoint
        self.endpoint =endpoint
        self.model = "doubao-1-5-pro-32k-250115"
    
    def check_completeness(self, text: str, return_detail: bool = False) -> Dict:
        """
        检查文本语义完整性
        
        Args:
            text: 待检测的中文文本
            return_detail: 是否返回详细分析
            
        Returns:
            dict: {
                'text': 原文本,
                'is_complete': 是否完整 (bool),
                'confidence': 置信度 (0-1),
                'reason': 判断原因,
                'suggestions': 改进建议 (可选)
            }
        """
        
        # 构建prompt
        if return_detail:
            prompt = f"""请分析以下中文文本的语义完整性。

                        文本："{text}"

                        请从以下维度判断：
                        1. 语法结构是否完整（主谓宾等成分）
                        2. 语义表达是否完整,大致能理解意思即可，不用很精准
                        3、不要根据标点符号判断完整不完整，只看句子内容，完全忽略标点符号

                        请严格按照以下JSON格式返回（不要有任何其他文字）：
                        {{
                            "is_complete": True或False,
                            "confidence": 0.0到1.0之间的数值,
                            "reason": "详细的判断理由",
                            "suggestions": "如果不完整，给出改进建议（可选）"
                        }}"""
        else:
            # 1. 语法结构是否完整（主谓宾等成分）
            # 2. 语义表达是否完整（有无明显截断）
            prompt = f"""请分析以下中文文本的语义完整性。

                        文本："{text}"

                        请从以下维度判断：
                        1. 语法结构是否完整（主谓宾等成分）
                        2. 语义表达是否完整,大致能理解意思即可，不用很精准
                        3、不要根据标点符号判断完整不完整，只看句子内容，完全忽略标点符号


                        请严格按照以下JSON格式返回（不要有任何其他文字）：
                        {{
                            "is_complete": true或false,
                            "confidence": 0.0到1.0之间的一个小数
                        }}"""
        
        # 构建请求
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个专业的中文语义分析助手，擅长判断句子的完整性。请始终以JSON格式返回结果。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,  # 低温度保证稳定输出
            "max_tokens": 500
        }
        
        try:
            # 发送请求
            response = requests.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            # 解析响应
            content = result['choices'][0]['message']['content']
            
            # 提取JSON（处理可能的markdown代码块）
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            analysis = json.loads(content)
            
            # 构建返回结果
            return {
                'text': text,
                'is_complete': analysis.get('is_complete', False),
                'confidence': float(analysis.get('confidence', 0.0)),
                'reason': analysis.get('reason', ''),
                'suggestions': analysis.get('suggestions', ''),
                'status': 'success'
            }
            
        except requests.exceptions.RequestException as e:
            return {
                'text': text,
                'is_complete': False,
                'confidence': 0.0,
                'reason': f'API请求失败: {str(e)}',
                'suggestions': '',
                'status': 'error'
            }
        except json.JSONDecodeError as e:
            return {
                'text': text,
                'is_complete': False,
                'confidence': 0.0,
                'reason': f'JSON解析失败: {str(e)}',
                'suggestions': '',
                'status': 'error',
                'raw_response': content if 'content' in locals() else ''
            }
        except Exception as e:
            return {
                'text': text,
                'is_complete': False,
                'confidence': 0.0,
                'reason': f'未知错误: {str(e)}',
                'suggestions': '',
                'status': 'error'
            }
    
    def batch_check(self, texts: list) -> list:
        """
        批量检测多个文本
        
        Args:
            texts: 文本列表
            
        Returns:
            list: 检测结果列表
        """
        results = []
        for text in texts:
            result = self.check_completeness(text)
            results.append(result)
        return results
