#!/usr/bin/env python3
"""
推理脚本资源评测工具
评测指标：
- CPU使用率
- GPU使用率（利用率、显存占用）
- 内存占用（RSS, VMS）
- 推理时间（总时间、平均每帧时间）
- FPS（帧率）
"""

import subprocess
import psutil
import time
import os
import sys
import json
import threading
from datetime import datetime
from pathlib import Path


class InferenceBenchmark:
    """推理脚本资源评测工具"""
    
    def __init__(self, script_path, config_path, checkpoint_path, 
                 gpu_id=0, additional_args=None, sample_interval=0.1, debug=False):
        """
        Args:
            script_path: 推理脚本路径（如 tools/test.py）
            config_path: 配置文件路径
            checkpoint_path: 模型检查点路径
            gpu_id: GPU ID（默认0）
            additional_args: 额外的命令行参数列表（如 ['--eval', 'bbox']）
            sample_interval: 采样间隔（秒），默认0.1秒
        """
        self.script_path = script_path
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.gpu_id = gpu_id
        self.additional_args = additional_args or []
        self.sample_interval = sample_interval
        self.debug = debug
        
        # 检查文件是否存在
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"推理脚本不存在: {script_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
        
        # 检查GPU是否可用
        self.gpu_available = self._check_gpu_available()
        if not self.gpu_available:
            print("⚠️  警告: 未检测到NVIDIA GPU或nvidia-smi不可用，将无法监控GPU使用情况")
    
    def _check_gpu_available(self):
        """检查GPU是否可用"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                gpu_indices = [int(x.strip()) for x in result.stdout.strip().split('\n') if x.strip()]
                if self.gpu_id not in gpu_indices:
                    print(f"⚠️  警告: GPU {self.gpu_id} 不存在，可用GPU: {gpu_indices}")
                    if gpu_indices:
                        self.gpu_id = gpu_indices[0]
                        print(f"   将使用GPU {self.gpu_id}")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass
        return False
    
    def get_gpu_stats(self, process_pid=None):
        """获取GPU使用统计"""
        if not self.gpu_available:
            return None
        
        try:
            # 查询GPU使用率和显存（使用更详细的查询）
            # 注意：utilization.gpu是过去1秒的平均值，可能不够实时
            # 我们同时查询多个指标以获得更准确的信息
            query = f'--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu'
            result = subprocess.run(
                ['nvidia-smi', query, '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5, check=False
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    parts = [x.strip() for x in line.split(',')]
                    if len(parts) >= 6 and int(parts[0]) == self.gpu_id:
                        gpu_util = float(parts[1])
                        mem_util = float(parts[2])
                        mem_used = float(parts[3])
                        mem_total = float(parts[4])
                        power_draw = float(parts[5]) if len(parts) > 5 and parts[5] else 0
                        temp = float(parts[6]) if len(parts) > 6 and parts[6] else 0
                        mem_percent = (mem_used / mem_total) * 100 if mem_total > 0 else 0
                        
                        stats = {
                            'gpu_utilization_percent': gpu_util,
                            'gpu_memory_used_mb': mem_used,
                            'gpu_memory_total_mb': mem_total,
                            'gpu_memory_percent': mem_percent,
                            'gpu_memory_utilization_percent': mem_util
                        }
                        
                        # 添加功耗和温度信息（可以帮助判断GPU是否在工作）
                        if power_draw > 0:
                            stats['gpu_power_watts'] = power_draw
                        if temp > 0:
                            stats['gpu_temperature_c'] = temp
                        
                        # 如果提供了进程PID，尝试查询该进程及其子进程的GPU使用情况
                        if process_pid is not None:
                            try:
                                # 获取进程及其所有子进程的PID
                                pids_to_check = [process_pid]
                                try:
                                    proc = psutil.Process(process_pid)
                                    for child in proc.children(recursive=True):
                                        pids_to_check.append(child.pid)
                                except (psutil.NoSuchProcess, psutil.AccessDenied):
                                    pass
                                
                                # 查询所有GPU进程的使用情况
                                proc_query = '--query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits'
                                proc_result = subprocess.run(
                                    ['nvidia-smi', proc_query],
                                    capture_output=True, text=True, timeout=3, check=False
                                )
                                if proc_result.returncode == 0:
                                    total_proc_mem = 0
                                    proc_count = 0
                                    # 查找匹配的进程
                                    for proc_line in proc_result.stdout.strip().split('\n'):
                                        if proc_line.strip():
                                            proc_parts = [x.strip() for x in proc_line.split(',')]
                                            if len(proc_parts) >= 2:
                                                try:
                                                    pid = int(proc_parts[0])
                                                    # 检查是否是目标进程或其子进程
                                                    if pid in pids_to_check:
                                                        proc_count += 1
                                                        if len(proc_parts) >= 3:
                                                            proc_mem = float(proc_parts[2])
                                                            total_proc_mem += proc_mem
                                                except ValueError:
                                                    continue
                                    if proc_count > 0:
                                        stats['process_gpu_memory_mb'] = total_proc_mem
                                        stats['process_gpu_count'] = proc_count
                            except Exception:
                                pass  # 忽略进程查询错误
                        
                        return stats
        except (subprocess.TimeoutExpired, ValueError, IndexError):
            pass
        
        return None
    
    def get_process_stats(self, process):
        """获取进程资源使用统计"""
        try:
            # CPU使用率
            cpu_percent = process.cpu_percent(interval=0.01)
            
            # 内存使用
            mem_info = process.memory_info()
            rss_mb = mem_info.rss / 1024 / 1024  # Resident Set Size (MB)
            vms_mb = mem_info.vms / 1024 / 1024  # Virtual Memory Size (MB)
            
            # 内存百分比
            mem_percent = process.memory_percent()
            
            # 获取子进程统计（Python脚本可能启动子进程）
            children_stats = []
            try:
                for child in process.children(recursive=True):
                    try:
                        child_mem = child.memory_info()
                        children_stats.append({
                            'rss_mb': child_mem.rss / 1024 / 1024,
                            'vms_mb': child_mem.vms / 1024 / 1024
                        })
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            
            # 计算子进程总内存
            children_rss = sum(c['rss_mb'] for c in children_stats)
            children_vms = sum(c['vms_mb'] for c in children_stats)
            
            stats = {
                'cpu_percent': cpu_percent,
                'memory_rss_mb': rss_mb,
                'memory_vms_mb': vms_mb,
                'memory_percent': mem_percent,
                'children_count': len(children_stats),
                'children_rss_mb': children_rss,
                'children_vms_mb': children_vms,
                'total_rss_mb': rss_mb + children_rss,
                'total_vms_mb': vms_mb + children_vms
            }
            
            # 添加GPU统计（传入进程PID以便更精确监控）
            gpu_stats = self.get_gpu_stats(process.pid)
            if gpu_stats:
                stats.update(gpu_stats)
            
            return stats
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None
    
    def run_benchmark(self):
        """运行性能评测"""
        print("=" * 70)
        print("推理脚本资源评测")
        print("=" * 70)
        print(f"推理脚本: {self.script_path}")
        print(f"配置文件: {self.config_path}")
        print(f"检查点文件: {self.checkpoint_path}")
        print(f"GPU ID: {self.gpu_id}")
        if self.additional_args:
            print(f"额外参数: {' '.join(self.additional_args)}")
        print(f"采样间隔: {self.sample_interval * 1000:.1f} ms")
        print("-" * 70)
        
        # 构建命令 - 使用当前Python解释器
        python_exe = sys.executable
        # 确保使用绝对路径
        script_abs = os.path.abspath(self.script_path)
        config_abs = os.path.abspath(self.config_path)
        checkpoint_abs = os.path.abspath(self.checkpoint_path)
        
        # 对于单GPU，直接运行test.py（已修改支持非分布式模式）
        # 不使用 torch.distributed.launch
        cmd = [python_exe, script_abs, config_abs, checkpoint_abs]
        cmd.extend(self.additional_args)
        
        # 设置环境变量
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)
        
        # 计算工作目录（项目根目录）
        # 从脚本路径向上查找项目根目录（包含 README.md 或 setup.py 的目录）
        script_dir = os.path.dirname(script_abs)
        work_dir = script_dir
        current_dir = script_dir
        
        # 向上查找项目根目录
        for _ in range(5):  # 最多向上查找5级
            # 检查是否是项目根目录（包含 README.md, setup.py, 或 projects/ 目录）
            if (os.path.exists(os.path.join(current_dir, 'README.md')) or
                os.path.exists(os.path.join(current_dir, 'README_zh.md')) or
                os.path.exists(os.path.join(current_dir, 'setup.py')) or
                os.path.exists(os.path.join(current_dir, 'projects'))):
                work_dir = current_dir
                break
            parent = os.path.dirname(current_dir)
            if parent == current_dir:  # 已到根目录
                break
            current_dir = parent
        
        # 如果没找到，使用脚本所在目录的父目录（假设脚本在 tools/ 下）
        if work_dir == script_dir and os.path.basename(script_dir) == 'tools':
            work_dir = os.path.dirname(script_dir)
        
        # 设置PYTHONPATH以便找到mmdet3d模块
        # 项目根目录需要添加到PYTHONPATH中
        project_root = work_dir
        current_pythonpath = env.get('PYTHONPATH', '')
        if current_pythonpath:
            # 如果PYTHONPATH已存在，将项目根目录添加到前面
            env['PYTHONPATH'] = f"{project_root}:{current_pythonpath}"
        else:
            env['PYTHONPATH'] = project_root
        
        # 记录GPU基线状态（在启动进程之前）
        gpu_baseline = None
        if self.gpu_available:
            print("正在记录GPU基线状态...")
            time.sleep(0.5)  # 等待0.5秒确保GPU状态稳定
            baseline_stats = self.get_gpu_stats()
            if baseline_stats:
                gpu_baseline = {
                    'memory_used_mb': baseline_stats.get('gpu_memory_used_mb', 0),
                    'utilization_percent': baseline_stats.get('gpu_utilization_percent', 0),
                    'power_watts': baseline_stats.get('gpu_power_watts', 0),
                    'temperature_c': baseline_stats.get('gpu_temperature_c', 0)
                }
                print(f"GPU基线状态: 显存={gpu_baseline['memory_used_mb']:.0f}MB, "
                      f"利用率={gpu_baseline['utilization_percent']:.1f}%")
        
        # 启动进程
        start_time = time.time()
        try:
            process = psutil.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=work_dir
            )
        except Exception as e:
            raise RuntimeError(f"启动推理脚本失败: {e}") from e
        
        # 监控资源使用
        stats_list = []
        monitoring = True
        
        def monitor_loop():
            """监控循环（在单独线程中运行）"""
            nonlocal monitoring
            # 立即开始监控，不等待（确保捕获GPU使用的初始阶段）
            sample_count = 0
            
            while monitoring and process.poll() is None:
                stats = self.get_process_stats(process)
                if stats:
                    # 计算GPU显存增量（减去基线）
                    if gpu_baseline and 'gpu_memory_used_mb' in stats:
                        baseline_mem = gpu_baseline.get('memory_used_mb', 0)
                        current_mem = stats.get('gpu_memory_used_mb', 0)
                        stats['gpu_memory_increment_mb'] = max(0, current_mem - baseline_mem)
                    
                    # 计算GPU利用率增量（减去基线）
                    if gpu_baseline and 'gpu_utilization_percent' in stats:
                        baseline_util = gpu_baseline.get('utilization_percent', 0)
                        current_util = stats.get('gpu_utilization_percent', 0)
                        stats['gpu_utilization_increment_percent'] = max(0, current_util - baseline_util)
                    
                    stats['timestamp'] = time.time() - start_time
                    stats_list.append(stats)
                    sample_count += 1
                    
                    # 调试模式：实时打印GPU使用情况
                    if self.debug and 'gpu_utilization_percent' in stats:
                        gpu_util = stats.get('gpu_utilization_percent', 0)
                        gpu_mem = stats.get('gpu_memory_used_mb', 0)
                        gpu_mem_inc = stats.get('gpu_memory_increment_mb', 0)
                        gpu_util_inc = stats.get('gpu_utilization_increment_percent', 0)
                        print(f"[{stats['timestamp']:.2f}s] GPU: {gpu_util:.1f}% (+{gpu_util_inc:.1f}%) | "
                              f"显存: {gpu_mem:.0f}MB (+{gpu_mem_inc:.0f}MB)", flush=True)
                    
                    # 前几次采样使用更短的间隔以快速捕获GPU使用变化
                    if sample_count <= 5:
                        time.sleep(0.01)  # 前5次采样使用10ms间隔
                    elif sample_count <= 20:
                        time.sleep(0.02)  # 接下来15次使用20ms间隔
                    else:
                        time.sleep(self.sample_interval)  # 之后使用正常间隔
                else:
                    time.sleep(self.sample_interval)
            
            # 获取最后一次统计
            if monitoring:
                stats = self.get_process_stats(process)
                if stats:
                    # 计算GPU显存增量（减去基线）
                    if gpu_baseline and 'gpu_memory_used_mb' in stats:
                        baseline_mem = gpu_baseline.get('memory_used_mb', 0)
                        current_mem = stats.get('gpu_memory_used_mb', 0)
                        stats['gpu_memory_increment_mb'] = max(0, current_mem - baseline_mem)
                    
                    # 计算GPU利用率增量（减去基线）
                    if gpu_baseline and 'gpu_utilization_percent' in stats:
                        baseline_util = gpu_baseline.get('utilization_percent', 0)
                        current_util = stats.get('gpu_utilization_percent', 0)
                        stats['gpu_utilization_increment_percent'] = max(0, current_util - baseline_util)
                    
                    stats['timestamp'] = time.time() - start_time
                    stats_list.append(stats)
        
        # 启动监控线程
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        
        try:
            # 等待进程结束并获取输出
            stdout, stderr = process.communicate()
            end_time = time.time()
            monitoring = False
            
            # 等待监控线程结束
            monitor_thread.join(timeout=1.0)
            
        except KeyboardInterrupt:
            print("\n评测被中断")
            monitoring = False
            process.terminate()
            try:
                process.wait(timeout=5)
            except psutil.TimeoutExpired:
                process.kill()
            sys.exit(1)
        
        # 计算总时间
        total_time = end_time - start_time
        
        # 解析输出获取处理样本数
        output_text = stdout.decode('utf-8', errors='ignore')
        error_text = stderr.decode('utf-8', errors='ignore')
        processed_samples = self._parse_sample_count(output_text, error_text)
        
        # 检测进程是否真正运行（通过GPU使用情况判断）
        gpu_actually_used = False
        if gpu_baseline and stats_list:
            # 检查是否有明显的GPU使用（利用率增量>10%或显存增量>50MB）
            max_util_inc = max([s.get('gpu_utilization_increment_percent', 0) for s in stats_list], default=0)
            max_mem_inc = max([s.get('gpu_memory_increment_mb', 0) for s in stats_list], default=0)
            if max_util_inc > 10 or max_mem_inc > 50:
                gpu_actually_used = True
        
        # 如果进程退出码非0且GPU使用很少，说明进程可能没有真正运行
        if process.returncode != 0 and not gpu_actually_used:
            print(f"\n⚠️  警告: 进程异常退出（退出码: {process.returncode}），且GPU使用很少")
            print(f"   这表明进程可能在启动阶段就失败了，没有真正运行推理任务")
            print(f"   GPU利用率增量: {max_util_inc:.1f}% (正常推理应该>20%)")
            print(f"   显存增量: {max_mem_inc:.0f}MB (正常推理应该>100MB)")
            if error_text:
                error_preview = error_text[-500:] if len(error_text) > 500 else error_text
                print(f"\n   错误信息预览:")
                for line in error_preview.split('\n')[-8:]:
                    if line.strip():
                        print(f"     {line}")
        
        # 验证：如果进程运行太快，采样可能不准确
        if total_time < 1.0 and len(stats_list) < 10:
            print(f"\n💡 提示: 进程运行较快 ({total_time:.3f}秒)，采样次数较少 ({len(stats_list)}次)")
            print(f"   建议: 处理更多样本以获得更准确的统计采样")
        
        # 检测常见错误
        errors_detected = self._detect_errors(error_text, process.returncode)
        
        # 计算统计指标
        results = self._compute_statistics(stats_list, total_time, processed_samples, gpu_baseline)
        
        # 添加进程运行状态信息
        results['process_status'] = {
            'return_code': process.returncode,
            'gpu_actually_used': gpu_actually_used,
            'total_time_seconds': total_time,
            'max_gpu_utilization_increment': max_util_inc if gpu_baseline else 0,
            'max_gpu_memory_increment_mb': max_mem_inc if gpu_baseline else 0
        }
        
        # 保存原始输出
        results['stdout'] = output_text
        results['stderr'] = error_text
        results['return_code'] = process.returncode
        if errors_detected:
            results['errors_detected'] = errors_detected
        
        return results
    
    def _parse_sample_count(self, output_text, error_text):
        """从输出中解析处理的样本数"""
        import re
        
        combined_text = output_text + '\n' + error_text
        
        # 尝试多种模式匹配样本数
        patterns = [
            # mmdet3d 进度条格式: [>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 6019/6019
            r'\[(>+\s*)\]\s*(\d+)/(\d+)',
            # Done: [1234/1234]
            r'Done.*?\[(\d+)/',
            # Evaluating 6019 samples
            r'(?:Evaluating|Processing)\s+(\d+)\s+samples',
            # load 6019 samples
            r'load(?:ed)?\s+(\d+)\s+samples',
            # 中文模式
            r'处理.*?(\d+).*?样本',
            r'评估.*?(\d+).*?样本',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            if matches:
                try:
                    # 处理不同的匹配组格式
                    if isinstance(matches[-1], tuple):
                        # 取元组中最后一个非空数字
                        for num in reversed(matches[-1]):
                            if num and num.isdigit():
                                return int(num)
                    else:
                        return int(matches[-1])
                except (ValueError, AttributeError):
                    continue
        
        # 如果解析失败，返回0并给出警告
        if output_text or error_text:
            print(f"\n⚠️  警告: 无法从输出中解析处理样本数")
            print(f"   输出预览（最后300字符）: {(output_text + error_text)[-300:]}")
        
        return 0
    
    def _detect_errors(self, error_text, return_code):
        """检测常见错误并返回错误信息"""
        errors = []
        
        if return_code != 0:
            # 检测评测指标错误
            if 'metric' in error_text.lower() and 'not supported' in error_text.lower():
                import re
                metric_match = re.search(r"metric ([\w]+) is not supported", error_text)
                if metric_match:
                    wrong_metric = metric_match.group(1)
                    errors.append({
                        'type': 'invalid_metric',
                        'message': f'不支持的评测指标: {wrong_metric}',
                        'description': 'MapTR 是地图重建任务，不支持目标检测的 bbox 指标',
                        'suggestions': [
                            '使用 chamfer 指标: --eval chamfer',
                            '或使用 iou 指标: --eval iou',
                            '查看配置文件了解支持的评测指标'
                        ]
                    })
            
            # 检测文件不存在错误
            if 'FileNotFoundError' in error_text or 'No such file or directory' in error_text:
                import re
                # 尝试提取文件路径
                match = re.search(r"(?:FileNotFoundError|No such file or directory)[:\s]+.*?['\"]([^'\"]+)['\"]", error_text)
                if match:
                    missing_file = match.group(1)
                    errors.append({
                        'type': 'file_not_found',
                        'message': f'数据文件不存在',
                        'description': f'找不到文件: {missing_file}',
                        'suggestions': [
                            '检查数据集是否已下载并放置在正确位置',
                            '如果是 nuscenes 数据集，请参考 docs/prepare_dataset.md 准备数据',
                            '检查配置文件中的数据路径设置是否正确',
                            f'确保文件存在: ls -la {missing_file}'
                        ]
                    })
                else:
                    errors.append({
                        'type': 'file_not_found',
                        'message': '数据文件不存在',
                        'description': '检测到文件不存在错误，请检查数据集是否正确配置',
                        'suggestions': [
                            '检查数据集是否已下载并放置在正确位置',
                            '参考 docs/prepare_dataset.md 准备数据',
                            '检查配置文件中的数据路径设置'
                        ]
                    })
            
            # 检测numba错误
            if 'numba.errors' in error_text or 'ModuleNotFoundError' in error_text and 'numba' in error_text:
                errors.append({
                    'type': 'numba_compatibility',
                    'message': 'Numba版本兼容性问题',
                    'description': '检测到numba.errors导入错误，这通常是numba版本不兼容导致的',
                    'suggestions': [
                        '检查numba版本: pip show numba',
                        '尝试降级numba: pip install numba==0.48.0',
                        '或者升级numba到最新版本: pip install --upgrade numba',
                        '如果使用numba >= 0.57，可能需要修改代码: from numba import NumbaPerformanceWarning (而不是from numba.errors)'
                    ]
                })
            
            # 检测其他常见错误
            if 'ModuleNotFoundError' in error_text and 'numba' not in error_text:
                import re
                match = re.search(r"ModuleNotFoundError: No module named '([^']+)'", error_text)
                if match:
                    module_name = match.group(1)
                    errors.append({
                        'type': 'missing_module',
                        'message': f'缺少模块: {module_name}',
                        'suggestions': [f'安装缺失的模块: pip install {module_name}']
                    })
            
            if 'CUDA' in error_text or 'cuda' in error_text:
                errors.append({
                    'type': 'cuda_error',
                    'message': 'CUDA相关错误',
                    'suggestions': [
                        '检查CUDA驱动: nvidia-smi',
                        '检查PyTorch CUDA支持: python -c "import torch; print(torch.cuda.is_available())"',
                        '检查CUDA版本兼容性'
                    ]
                })
            
            # 检测系统库版本错误（libstdc++, CXXABI等）
            if 'libstdc++' in error_text or 'CXXABI' in error_text or 'version `' in error_text:
                errors.append({
                    'type': 'system_library_error',
                    'message': '系统库版本兼容性问题',
                    'description': '检测到系统库版本不兼容错误（如libstdc++、CXXABI等），这通常是conda环境与系统库版本不匹配导致的',
                    'suggestions': [
                        '尝试使用conda环境中的libstdc++: conda install -c conda-forge libstdcxx-ng',
                        '或者更新系统库: sudo apt-get update && sudo apt-get install libstdc++6',
                        '检查conda环境: conda list | grep libstdc',
                        '如果使用conda环境，尝试: conda update --all',
                        '或者设置LD_LIBRARY_PATH指向conda环境的lib目录'
                    ]
                })
        
        return errors if errors else None
    
    def _compute_statistics(self, stats_list, total_time, processed_samples, gpu_baseline=None):
        """计算统计指标"""
        if not stats_list:
            return {
                'error': '未能收集到性能数据',
                'total_time': total_time,
                'processed_samples': processed_samples
            }
        
        # CPU统计
        cpu_values = [s['cpu_percent'] for s in stats_list if 'cpu_percent' in s]
        cpu_mean = sum(cpu_values) / len(cpu_values) if cpu_values else 0
        cpu_max = max(cpu_values) if cpu_values else 0
        cpu_min = min(cpu_values) if cpu_values else 0
        
        # 内存统计 (RSS)
        rss_values = [s.get('total_rss_mb', s.get('memory_rss_mb', 0)) for s in stats_list]
        rss_mean = sum(rss_values) / len(rss_values) if rss_values else 0
        rss_max = max(rss_values) if rss_values else 0
        rss_min = min(rss_values) if rss_values else 0
        
        # 内存统计 (VMS)
        vms_values = [s.get('total_vms_mb', s.get('memory_vms_mb', 0)) for s in stats_list]
        vms_mean = sum(vms_values) / len(vms_values) if vms_values else 0
        vms_max = max(vms_values) if vms_values else 0
        
        # 内存百分比
        mem_percent_values = [s['memory_percent'] for s in stats_list if 'memory_percent' in s]
        mem_percent_mean = sum(mem_percent_values) / len(mem_percent_values) if mem_percent_values else 0
        mem_percent_max = max(mem_percent_values) if mem_percent_values else 0
        
        # GPU统计
        gpu_stats = {}
        if self.gpu_available:
            gpu_util_values = [s['gpu_utilization_percent'] for s in stats_list if 'gpu_utilization_percent' in s]
            gpu_mem_values = [s['gpu_memory_used_mb'] for s in stats_list if 'gpu_memory_used_mb' in s]
            gpu_mem_percent_values = [s['gpu_memory_percent'] for s in stats_list if 'gpu_memory_percent' in s]
            gpu_power_values = [s['gpu_power_watts'] for s in stats_list if 'gpu_power_watts' in s]
            gpu_temp_values = [s['gpu_temperature_c'] for s in stats_list if 'gpu_temperature_c' in s]
            
            # 计算增量（减去基线）
            gpu_util_inc_values = [s.get('gpu_utilization_increment_percent', 0) for s in stats_list if 'gpu_utilization_increment_percent' in s]
            gpu_mem_inc_values = [s.get('gpu_memory_increment_mb', 0) for s in stats_list if 'gpu_memory_increment_mb' in s]
            
            if gpu_util_values:
                gpu_stats = {
                    'utilization': {
                        'mean_percent': round(sum(gpu_util_values) / len(gpu_util_values), 2),
                        'max_percent': round(max(gpu_util_values), 2),
                        'min_percent': round(min(gpu_util_values), 2),
                    },
                    'memory': {
                        'mean_mb': round(sum(gpu_mem_values) / len(gpu_mem_values), 2),
                        'max_mb': round(max(gpu_mem_values), 2),
                        'min_mb': round(min(gpu_mem_values), 2),
                    },
                    'memory_percent': {
                        'mean': round(sum(gpu_mem_percent_values) / len(gpu_mem_percent_values), 2),
                        'max': round(max(gpu_mem_percent_values), 2),
                    }
                }
                
                # 添加增量统计（减去基线后的实际使用）
                if gpu_baseline:
                    gpu_stats['baseline'] = {
                        'memory_mb': round(gpu_baseline.get('memory_used_mb', 0), 2),
                        'utilization_percent': round(gpu_baseline.get('utilization_percent', 0), 2)
                    }
                    
                    if gpu_util_inc_values:
                        gpu_stats['utilization_increment'] = {
                            'mean_percent': round(sum(gpu_util_inc_values) / len(gpu_util_inc_values), 2),
                            'max_percent': round(max(gpu_util_inc_values), 2),
                            'min_percent': round(min(gpu_util_inc_values), 2),
                        }
                    
                    if gpu_mem_inc_values:
                        gpu_stats['memory_increment'] = {
                            'mean_mb': round(sum(gpu_mem_inc_values) / len(gpu_mem_inc_values), 2),
                            'max_mb': round(max(gpu_mem_inc_values), 2),
                            'min_mb': round(min(gpu_mem_inc_values), 2),
                        }
                
                # 获取GPU总显存
                if stats_list and 'gpu_memory_total_mb' in stats_list[0]:
                    gpu_stats['memory']['total_mb'] = round(stats_list[0]['gpu_memory_total_mb'], 2)
                
                # 添加功耗和温度统计
                if gpu_power_values:
                    gpu_stats['power'] = {
                        'mean_watts': round(sum(gpu_power_values) / len(gpu_power_values), 2),
                        'max_watts': round(max(gpu_power_values), 2),
                        'min_watts': round(min(gpu_power_values), 2),
                    }
                    if gpu_baseline and 'power_watts' in gpu_baseline:
                        baseline_power = gpu_baseline.get('power_watts', 0)
                        gpu_stats['power']['baseline_watts'] = round(baseline_power, 2)
                        gpu_stats['power']['increment_mean_watts'] = round(
                            sum(gpu_power_values) / len(gpu_power_values) - baseline_power, 2)
                        gpu_stats['power']['increment_max_watts'] = round(
                            max(gpu_power_values) - baseline_power, 2)
                
                if gpu_temp_values:
                    gpu_stats['temperature'] = {
                        'mean_c': round(sum(gpu_temp_values) / len(gpu_temp_values), 2),
                        'max_c': round(max(gpu_temp_values), 2),
                        'min_c': round(min(gpu_temp_values), 2),
                    }
        
        # 计算FPS和时间
        if total_time > 0 and processed_samples > 0:
            fps = processed_samples / total_time
            time_per_sample = total_time / processed_samples
        else:
            fps = 0
            time_per_sample = 0
        
        results = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config': {
                'script': self.script_path,
                'config': self.config_path,
                'checkpoint': self.checkpoint_path,
                'gpu_id': self.gpu_id,
                'additional_args': self.additional_args
            },
            'performance': {
                'total_time_seconds': round(total_time, 3),
                'processed_samples': processed_samples,
                'fps': round(fps, 2),
                'time_per_sample_ms': round(time_per_sample * 1000, 2),
            },
            'cpu': {
                'mean_percent': round(cpu_mean, 2),
                'max_percent': round(cpu_max, 2),
                'min_percent': round(cpu_min, 2),
            },
            'memory': {
                'rss': {
                    'mean_mb': round(rss_mean, 2),
                    'max_mb': round(rss_max, 2),
                    'min_mb': round(rss_min, 2),
                },
                'vms': {
                    'mean_mb': round(vms_mean, 2),
                    'max_mb': round(vms_max, 2),
                },
                'percent': {
                    'mean': round(mem_percent_mean, 2),
                    'max': round(mem_percent_max, 2),
                }
            },
            'samples': len(stats_list),
            'sample_interval_ms': round(self.sample_interval * 1000, 1)
        }
        
        # 添加GPU统计
        if gpu_stats:
            results['gpu'] = gpu_stats
        
        return results
    
    def print_results(self, results):
        """打印评测结果"""
        if 'error' in results:
            print(f"\n错误: {results['error']}")
            return
        
        # 检查进程运行状态
        if 'process_status' in results:
            status = results['process_status']
            if status['return_code'] != 0 and not status.get('gpu_actually_used', False):
                print("\n" + "=" * 70)
                print("⚠️  进程未正常运行")
                print("=" * 70)
                print(f"  退出码: {status['return_code']}")
                print(f"  运行时间: {status['total_time_seconds']:.3f} 秒")
                print(f"  GPU实际使用: {'是' if status.get('gpu_actually_used') else '否'}")
                print("\n  进程可能在启动阶段就失败了，没有真正运行推理任务。")
                print("  请先解决依赖问题（如上面的numba错误），然后重新运行。")
                print("=" * 70 + "\n")
        
        # 如果有检测到的错误，先显示错误信息
        if 'errors_detected' in results and results['errors_detected']:
            print("\n" + "=" * 70)
            print("⚠️  检测到错误")
            print("=" * 70)
            for error in results['errors_detected']:
                print(f"\n【{error['message']}】")
                if 'description' in error:
                    print(f"  {error['description']}")
                if 'suggestions' in error:
                    print("  建议解决方案:")
                    for suggestion in error['suggestions']:
                        print(f"    • {suggestion}")
            print("\n" + "=" * 70)
        
        print("\n" + "=" * 70)
        print("评测结果")
        print("=" * 70)
        
        # 性能指标
        print("\n【性能指标】")
        perf = results['performance']
        print(f"  总运行时间:     {perf['total_time_seconds']:.3f} 秒")
        print(f"  处理样本数:     {perf['processed_samples']} 个")
        print(f"  平均吞吐量:     {perf['fps']:.2f} samples/s")
        print(f"  平均每样本时间: {perf['time_per_sample_ms']:.2f} ms")
        
        # CPU使用率
        print("\n【CPU使用率】")
        cpu = results['cpu']
        print(f"  平均:           {cpu['mean_percent']:.2f}%")
        print(f"  峰值:           {cpu['max_percent']:.2f}%")
        print(f"  最低:           {cpu['min_percent']:.2f}%")
        
        # 内存使用
        print("\n【内存使用】")
        mem = results['memory']
        print(f"  RSS (实际物理内存):")
        print(f"    平均:         {mem['rss']['mean_mb']:.2f} MB")
        print(f"    峰值:         {mem['rss']['max_mb']:.2f} MB")
        print(f"    最低:         {mem['rss']['min_mb']:.2f} MB")
        print(f"  VMS (虚拟内存):")
        print(f"    平均:         {mem['vms']['mean_mb']:.2f} MB")
        print(f"    峰值:         {mem['vms']['max_mb']:.2f} MB")
        print(f"  内存占用率:")
        print(f"    平均:         {mem['percent']['mean']:.2f}%")
        print(f"    峰值:         {mem['percent']['max']:.2f}%")
        
        # GPU使用
        if 'gpu' in results:
            print("\n【GPU使用】")
            gpu = results['gpu']
            print(f"  GPU利用率:")
            print(f"    平均:         {gpu['utilization']['mean_percent']:.2f}%")
            print(f"    峰值:         {gpu['utilization']['max_percent']:.2f}%")
            print(f"    最低:         {gpu['utilization']['min_percent']:.2f}%")
            print(f"  显存使用:")
            if 'total_mb' in gpu['memory']:
                print(f"    总显存:       {gpu['memory']['total_mb']:.2f} MB")
            print(f"    平均:         {gpu['memory']['mean_mb']:.2f} MB")
            print(f"    峰值:         {gpu['memory']['max_mb']:.2f} MB")
            print(f"    最低:         {gpu['memory']['min_mb']:.2f} MB")
            print(f"  显存占用率:")
            print(f"    平均:         {gpu['memory_percent']['mean']:.2f}%")
            print(f"    峰值:         {gpu['memory_percent']['max']:.2f}%")
            
            # 显示基线信息
            if 'baseline' in gpu:
                print(f"\n  GPU基线状态:")
                print(f"    显存基线:     {gpu['baseline']['memory_mb']:.2f} MB")
                print(f"    利用率基线:   {gpu['baseline']['utilization_percent']:.2f}%")
            
            # 显示增量（实际使用）
            if 'memory_increment' in gpu:
                print(f"\n  显存增量（实际使用）:")
                print(f"    平均:         {gpu['memory_increment']['mean_mb']:.2f} MB")
                print(f"    峰值:         {gpu['memory_increment']['max_mb']:.2f} MB")
                print(f"    最低:         {gpu['memory_increment']['min_mb']:.2f} MB")
            
            if 'utilization_increment' in gpu:
                print(f"  GPU利用率增量（实际使用）:")
                print(f"    平均:         {gpu['utilization_increment']['mean_percent']:.2f}%")
                print(f"    峰值:         {gpu['utilization_increment']['max_percent']:.2f}%")
                print(f"    最低:         {gpu['utilization_increment']['min_percent']:.2f}%")
            
            # 显示功耗和温度（如果可用）
            if 'power' in gpu:
                print(f"  GPU功耗:")
                if 'baseline_watts' in gpu['power']:
                    print(f"    基线:         {gpu['power']['baseline_watts']:.2f} W")
                    if 'increment_mean_watts' in gpu['power']:
                        print(f"    增量平均:     {gpu['power']['increment_mean_watts']:.2f} W")
                        print(f"    增量峰值:     {gpu['power']['increment_max_watts']:.2f} W")
                print(f"    平均:         {gpu['power']['mean_watts']:.2f} W")
                print(f"    峰值:         {gpu['power']['max_watts']:.2f} W")
                print(f"    最低:         {gpu['power']['min_watts']:.2f} W")
            if 'temperature' in gpu:
                print(f"  GPU温度:")
                print(f"    平均:         {gpu['temperature']['mean_c']:.2f} °C")
                print(f"    峰值:         {gpu['temperature']['max_c']:.2f} °C")
                print(f"    最低:         {gpu['temperature']['min_c']:.2f} °C")
            
            # 如果GPU利用率变化很小，给出提示
            util_range = gpu['utilization']['max_percent'] - gpu['utilization']['min_percent']
            if util_range < 5 and gpu['utilization']['mean_percent'] < 50:
                print(f"\n  💡 提示: GPU利用率变化较小 ({util_range:.2f}%)，可能原因：")
                print(f"     - 进程运行时间较短，GPU使用高峰期被错过")
                print(f"     - nvidia-smi的utilization.gpu是过去1秒的平均值，可能不够实时")
                print(f"     - 建议使用更短的采样间隔（--sample-interval 0.01）")
        else:
            print("\n【GPU使用】")
            print("  ⚠️  未收集到GPU数据（可能未使用GPU或nvidia-smi不可用）")
        
        # 采样信息
        print("\n【监控信息】")
        print(f"  采样次数:       {results['samples']}")
        print(f"  采样间隔:       {results['sample_interval_ms']} ms")
        
        # 如果采样次数太少，给出警告
        if results['samples'] < 10:
            print(f"  ⚠️  警告: 采样次数较少，统计数据可能不够准确")
            print(f"     建议: 处理更多样本以获得更准确的性能数据")
        
        print("\n" + "=" * 70)
    
    def save_results(self, results, output_file='benchmark_inference_results.json'):
        """保存评测结果到文件"""
        # 移除stdout/stderr以减小文件大小（可选）
        results_to_save = results.copy()
        if 'stdout' in results_to_save:
            results_to_save['stdout_length'] = len(results_to_save['stdout'])
            del results_to_save['stdout']
        if 'stderr' in results_to_save:
            results_to_save['stderr_length'] = len(results_to_save['stderr'])
            del results_to_save['stderr']
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_to_save, f, indent=2, ensure_ascii=False)
        
        print(f"\n评测结果已保存到: {output_file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='推理脚本资源评测工具')
    parser.add_argument('script', help='推理脚本路径（如 tools/test.py）')
    parser.add_argument('config', help='配置文件路径')
    parser.add_argument('checkpoint', help='检查点文件路径')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU ID (默认: 0)')
    parser.add_argument('--eval', type=str, nargs='+',
                        help='评估指标（如 bbox）')
    parser.add_argument('--out', type=str,
                        help='输出结果文件路径')
    parser.add_argument('--show', action='store_true',
                        help='显示结果')
    parser.add_argument('--show-dir', type=str,
                        help='结果保存目录')
    parser.add_argument('--fuse-conv-bn', action='store_true',
                        help='融合conv和bn层')
    parser.add_argument('--sample-interval', type=float, default=0.1,
                        help='采样间隔（秒，默认: 0.1）')
    parser.add_argument('--output', default='benchmark_inference_results.json',
                        help='输出JSON文件路径 (默认: benchmark_inference_results.json)')
    parser.add_argument('--no-save', action='store_true',
                        help='不保存结果到文件')
    parser.add_argument('--additional-args', nargs=argparse.REMAINDER,
                        help='额外的命令行参数（放在最后）')
    parser.add_argument('--debug', action='store_true',
                        help='调试模式：实时打印GPU使用情况')
    
    args = parser.parse_args()
    
    # 构建额外参数列表
    additional_args = []
    if args.eval:
        additional_args.extend(['--eval'] + args.eval)
    if args.out:
        additional_args.extend(['--out', args.out])
    if args.show:
        additional_args.append('--show')
    if args.show_dir:
        additional_args.extend(['--show-dir', args.show_dir])
    if args.fuse_conv_bn:
        additional_args.append('--fuse-conv-bn')
    if args.additional_args:
        additional_args.extend(args.additional_args)
    
    try:
        # 创建评测工具
        benchmark = InferenceBenchmark(
            script_path=args.script,
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            gpu_id=args.gpu_id,
            additional_args=additional_args,
            sample_interval=args.sample_interval,
            debug=args.debug
        )
        
        # 运行评测
        results = benchmark.run_benchmark()
        
        # 打印结果
        benchmark.print_results(results)
        
        # 保存结果
        if not args.no_save:
            benchmark.save_results(results, args.output)
        
        # 返回状态码
        return results.get('return_code', 0)
        
    except FileNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"评测过程中发生错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

