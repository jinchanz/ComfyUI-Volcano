def import_or_install(package_name):
    try:
        __import__(package_name)
        return True
    except ImportError:
        import pip
        pip.main(['install', package_name])
        return True

import_or_install('volcengine')

"""
ComfyUI-Volcano 自定义节点包
提供火山引擎文本到图像生成功能
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
