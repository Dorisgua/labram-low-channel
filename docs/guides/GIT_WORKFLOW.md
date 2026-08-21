# LaBraM Git / GitHub 操作手册

本文件保存可重复使用的 Git 操作方法。实际执行过的命令及结果记录在 `docs/COMMAND_LOG.md`。

## 1. 连接自己的 GitHub 仓库（首次一次）

先在 GitHub 新建一个空仓库，再执行：

```powershell
git -C D:\Documents\labram remote add origin https://github.com/用户名/仓库名.git
git -C D:\Documents\labram remote -v
```

不要照抄示例地址，必须替换成自己的仓库地址。

## 2. 第一次提交和上传

```powershell
git -C D:\Documents\labram status --short
git -C D:\Documents\labram add .
git -C D:\Documents\labram status --short
git -C D:\Documents\labram commit -m "chore: initialize LaBraM extension project"
git -C D:\Documents\labram push -u origin main
```

上传前必须先看两次 `status --short`，确认没有数据、模型、实验输出或官方 `upstream_labram` 被加入。

## 3. 日常修改和上传

```powershell
git -C D:\Documents\labram status --short
git -C D:\Documents\labram diff
git -C D:\Documents\labram add 路径
git -C D:\Documents\labram commit -m "说明本次修改"
git -C D:\Documents\labram push
```

优先指定文件或目录，不要在没有检查状态时直接提交全部内容。

## 4. 创建并上传 tag

适合实验基线、稳定版本或论文结果：

```powershell
git -C D:\Documents\labram tag -a v0.1.0 -m "first reproducible baseline"
git -C D:\Documents\labram push origin v0.1.0
```

查看 tag：

```powershell
git -C D:\Documents\labram tag --list
git -C D:\Documents\labram show v0.1.0
```

tag 名称和说明应按实际版本修改。

## 5. 快速查看历史

```powershell
git -C D:\Documents\labram log --oneline --decorate --graph --all -20
```

查看某次提交内容：

```powershell
git -C D:\Documents\labram show COMMIT_ID
```

## 6. 安全回溯

临时查看旧版本，不改写历史：

```powershell
git -C D:\Documents\labram switch --detach COMMIT_ID
```

返回主分支：

```powershell
git -C D:\Documents\labram switch main
```

撤销一个已经提交并可能上传的版本，推荐创建反向提交：

```powershell
git -C D:\Documents\labram revert COMMIT_ID
git -C D:\Documents\labram push
```

丢弃某个文件尚未提交的修改：

```powershell
git -C D:\Documents\labram restore 文件路径
```

`git reset --hard`、强制推送和删除 tag 都可能造成数据或历史丢失。不要直接执行，先确认目标和备份状态，并把操作记入 `COMMAND_LOG.md`。

## 7. 与官方 LaBraM 的关系

官方代码固定保存在：

```text
D:\Documents\labram\upstream_labram
```

自己的根仓库通过 `.gitignore` 排除 `upstream_labram/`，不重复上传官方完整仓库。当前记录的官方基准 commit：

```text
c431221e6cfd23dbfa9950e0180682fb322b0548
```
