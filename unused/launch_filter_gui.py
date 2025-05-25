#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
闭环优筛窗口启动脚本 - 修正版

快速启动交互式闭环优筛工具
"""

import os
import sys
import subprocess

def check_dependencies():
    """检查依赖库"""
    required_packages = {
        'tkinter': '图形界面库',
        'matplotlib': '绘图库', 
        'pandas': '数据处理库',
        'numpy': '数值计算库',
        'seaborn': '统计可视化库'
    }
    
    missing_packages = []
    
    for package, desc in required_packages.items():
        try:
            if package == 'tkinter':
                import tkinter
            else:
                __import__(package)
            print(f"✓ {package} ({desc}) - 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} ({desc}) - 未安装")
    
    return missing_packages

def install_packages(packages):
    """安装缺失的包"""
    print(f"\n正在安装缺失的依赖库...")
    for package in packages:
        if package == 'tkinter':
            print("注意: tkinter通常随Python一起安装，如果缺失请重新安装Python")
            continue
        
        print(f"安装 {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} 安装成功")
        except subprocess.CalledProcessError:
            print(f"✗ {package} 安装失败")
            return False
    
    return True

def create_directories():
    """创建必要的目录"""
    dirs = [
        "outputs",
        "outputs/loop_filter",
        "outputs/loop_analysis", 
        "outputs/loop_results",
        "outputs/visualizations"
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ 目录创建: {directory}")

def check_gui_file():
    """检查GUI文件是否存在"""
    gui_files = [
        "code/loop_filter_gui.py",
        "loop_filter_gui.py"
    ]
    
    for file_path in gui_files:
        if os.path.exists(file_path):
            return file_path
    
    return None

def create_gui_file():
    """创建GUI文件的提示信息"""
    print("\n" + "="*60)
    print("🚨 重要提示：需要创建GUI主程序文件")
    print("="*60)
    print("\n请按以下步骤操作：")
    print("\n1. 在 'code' 目录下创建文件: loop_filter_gui.py")
    print("2. 将完整的GUI代码复制到该文件中")
    print("3. 保存文件后重新运行此启动脚本")
    print("\n代码获取方式：")
    print("- 从之前的AI对话中复制完整的GUI代码")
    print("- 或者联系技术支持获取完整代码文件")
    print("\n" + "="*60)

def launch_gui():
    """启动GUI应用"""
    print("\n" + "="*60)
    print("启动物产中大图风控系统 - 交互式闭环优筛窗口")
    print("="*60)
    
    # 检查GUI文件
    gui_file = check_gui_file()
    if not gui_file:
        create_gui_file()
        return False
    
    print(f"✓ 找到GUI文件: {gui_file}")
    
    try:
        # 添加当前目录到Python路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # 添加code目录到Python路径
        code_dir = os.path.join(current_dir, 'code')
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)
        
        # 启动GUI
        print("正在启动GUI...")
        if gui_file.startswith('code/'):
            # 如果在code目录，先切换目录
            original_dir = os.getcwd()
            os.chdir('code')
            try:
                subprocess.run([sys.executable, 'loop_filter_gui.py'], check=True)
            finally:
                os.chdir(original_dir)
        else:
            # 直接运行
            subprocess.run([sys.executable, gui_file], check=True)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"GUI启动失败，错误代码: {e.returncode}")
        print("可能的原因：")
        print("1. 缺少依赖库")
        print("2. GUI代码有语法错误")
        print("3. 文件路径问题")
        return False
    except Exception as e:
        print(f"启动GUI时出错: {e}")
        return False

def main():
    """主函数"""
    print("物产中大图风控系统 - 优筛窗口启动器 (修正版)")
    print("="*55)
    
    # 1. 检查依赖
    print("\n1. 检查依赖库...")
    missing = check_dependencies()
    
    if missing:
        print(f"\n发现缺失依赖: {', '.join(missing)}")
        choice = input("是否自动安装缺失的依赖库? (y/n): ").lower()
        
        if choice == 'y':
            if not install_packages(missing):
                print("依赖库安装失败，程序退出")
                input("按回车键退出...")
                return
        else:
            print("请手动安装缺失的依赖库后重新运行")
            print("安装命令:")
            for pkg in missing:
                if pkg != 'tkinter':
                    print(f"  pip install {pkg}")
            input("按回车键退出...")
            return
    
    # 2. 创建目录
    print("\n2. 创建必要目录...")
    create_directories()
    
    # 3. 启动GUI
    print("\n3. 启动优筛窗口...")
    if not launch_gui():
        print("GUI启动失败")
        input("按回车键退出...")
        return
    
    print("\n程序已退出")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断程序")
    except Exception as e:
        print(f"\n程序出错: {e}")
        input("按回车键退出...")