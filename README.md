# Shian Codex Pet

`诗岸` 是一个可直接安装到本地 Codex app 的自定义宠物项目。

仓库里已经包含：

- 最终安装包：`package/pet.json`、`package/avatar.json`、`package/spritesheet.webp`
- 构建脚本：`scripts/build_reference_spritesheet.py`
- 一键安装脚本：
  - macOS / Linux: `scripts/install_pet.sh`
  - Windows PowerShell: `scripts/install_pet.ps1`
  - 跨平台 Python: `scripts/install_pet.py`

## 当前动作映射

当前构建输出使用固定 `8 x 9` atlas，尺寸 `1536 x 1872`，动作行如下：

| 行 | 状态 | 帧数 |
| --- | --- | --- |
| 0 | `idle` | 8 |
| 1 | `running-right` | 8 |
| 2 | `running-left` | 8 |
| 3 | `waving` | 4 |
| 4 | `jumping` | 5 |
| 5 | `failed` | 8 |
| 6 | `waiting` | 6 |
| 7 | `running` | 8 |
| 8 | `review` | 6 |

语义约定：

- `running-right` / `running-left`：拖拽方向移动
- `running`：Codex 正在工作
- `waiting`：等待用户输入或批准
- `review`：查看或检查结果

注意：这个项目只能控制每个状态“长什么样”，不能修改 Codex app 内部何时切到哪个状态。
通常要等 Codex thread 自己进入对应状态后，宠物才会切到 `running`、`waiting` 或 `review`；单纯在编辑器里手动敲代码，不一定会触发 `running`。

## 快速开始

### 依赖

需要本地有 Python 3。

安装构建依赖：

```bash
python3 -m pip install -r requirements.txt
```

Windows 也可以用：

```powershell
py -3 -m pip install -r requirements.txt
```

## 直接安装当前包

### macOS / Linux

```bash
bash scripts/install_pet.sh
```

### Windows PowerShell

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_pet.ps1
```

### 跨平台 Python

```bash
python3 scripts/install_pet.py
```

默认会安装到：

- macOS / Linux: `~/.codex/pets/shian-helper`
- Windows: `%USERPROFILE%\.codex\pets\shian-helper`

## 重新构建并安装

### macOS / Linux

```bash
bash scripts/install_pet.sh --build
```

### Windows PowerShell

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_pet.ps1 -Build
```

### 跨平台 Python

```bash
python3 scripts/install_pet.py --build
```

## 刷新 Codex app

如果你之前已经加载过这个宠物，覆盖本地文件后请在 Codex app 里刷新 custom pets，再重新选择 `诗岸`。

如果画面还没有更新，先确认以下路径里已经有新文件：

- `package/spritesheet.webp`
- `~/.codex/pets/shian-helper/spritesheet.webp`

## 仓库结构

```text
shian_pet/
  package/                  # 最终安装包
  references/               # 原始参考图和动作模板
  scripts/build_reference_spritesheet.py
  scripts/install_pet.py
  scripts/install_pet.sh
  scripts/install_pet.ps1
  PET_BRIEF.md              # 角色和动作约束
```

## 发布建议

推荐把以下内容提交到 GitHub：

- `package/`
- `references/`
- `scripts/`
- `PET_BRIEF.md`
- `requirements.txt`

不需要提交：

- `runs/`
- `backups/`
- `__pycache__/`
