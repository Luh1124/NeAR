import os
import pandas as pd
from subprocess import run, CalledProcessError
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def main():
    # 1. 读取并合并数据，获取需要传输的目录路径
    df1 = pd.read_csv('train_shade_slat_to_even_slat.csv')
    df2 = pd.read_csv('val_shade_slat_to_even_slat.csv')
    
    # 检查数据是否为空
    if df1.empty:
        print("警告: train_shade_slat_to_even_slat.csv 文件为空")
    if df2.empty:
        print("警告: val_shade_slat_to_even_slat.csv 文件为空")
    
    df = pd.concat([df1, df2], ignore_index=True)
    
    if df.empty:
        print("错误: 两个CSV文件都为空，没有数据可以处理")
        return
    
    # 提取目录路径（去重，避免重复传输同一目录）
    relight_paths = df['relight_image_path'].tolist()
    need_paths = list(set([os.path.dirname(path) for path in relight_paths]))  # 去重优化
    print(f"共需传输 {len(need_paths)} 个目录")

    # 2. 定义单个目录的传输函数（含结果捕获）
    def transfer_directory(dir_path):
        """
        执行单个目录的 tosutil 上传任务
        :param dir_path: 本地目录路径
        :return: (dir_path, 执行状态, 输出日志)
        """
        tos_target = 'tos://lhtest/3diclight/3diclight_even_8w9/8w9_neural_light_v6/'
        args = [
            '/baai-cwm-vepfs/cwm/hong.li/tosutil',
            'cp',
            '-r',          # 递归传输目录
            dir_path,      # 本地源目录
            tos_target,    # TOS 目标路径
            '-u',          # 仅上传更新的文件（增量传输）
        ]
        
        try:
            # 执行命令并捕获输出（stdout/stderr 合并为字符串）
            result = run(args, check=True, capture_output=True, text=True)
            return (dir_path, "成功", result.stdout)
        except CalledProcessError as e:
            # 捕获命令执行失败错误
            error_msg = f"返回码: {e.returncode}\n标准输出: {e.stdout}\n错误输出: {e.stderr}"
            return (dir_path, "失败", error_msg)
        except Exception as e:
            # 捕获其他错误（如目录不存在等）
            return (dir_path, "失败", str(e))

    # 3. 多线程并行执行（设置最大并行数为 8，可根据服务器性能调整）
    max_workers = 12
    print(f"启动 {max_workers} 个线程并行传输...")
    
    # 统计结果
    success_count = 0
    fail_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务到线程池
        future_tasks = {executor.submit(transfer_directory, path): path for path in need_paths}
        
        # 使用 tqdm 显示进度条
        with tqdm(total=len(need_paths), desc="上传进度", unit="个目录") as pbar:
            # 实时获取任务结果并打印
            for future in as_completed(future_tasks):
                dir_path, status, log = future.result()
                
                # 更新统计
                if status == "成功":
                    success_count += 1
                else:
                    fail_count += 1
                
                # 更新进度条描述
                pbar.set_postfix({
                    "成功": success_count,
                    "失败": fail_count,
                    "当前": os.path.basename(dir_path)[:30]
                })
                pbar.update(1)
                
                # 打印详细信息（失败的重点显示）
                if status == "失败":
                    tqdm.write(f"❌ 失败: {dir_path}")
                    tqdm.write(f"   错误: {log[:200]}...")
    
    # 最终统计
    print(f"\n{'='*50}")
    print(f"传输完成！总计: {len(need_paths)} 个目录")
    print(f"✅ 成功: {success_count} 个")
    print(f"❌ 失败: {fail_count} 个")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()