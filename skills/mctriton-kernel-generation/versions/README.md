# 版本分层 skill 维护约定

mcTriton 的 kernel 生成能力**随版本变化**（不同 triton 大版本 API 有差异，不同 SDK 小版本行为有差异）。
因此本目录按两层维护版本专属指导，平台会根据任务选择的 whl 包自动拼装：

```
SKILL.md                          # 基础技能（版本无关）
references/                       # 基础参考资料（版本无关）
versions/
  manifest.json                   # whl 版本 → {triton 大版本, SDK 小版本} 映射（新增版本在此登记）
  triton-<major>.md               # triton 大版本差异（如 triton-3.6.md / triton-3.0.md）
  sdk-<minor>.md                  # SDK 小版本差异（如 sdk-3.7.1.md / sdk-3.7.2.md）
```

系统 prompt 拼装顺序：`SKILL.md` → `references/*.md` → `versions/triton-<major>.md` → `versions/sdk-<minor>.md`。

## 新增一个 whl 版本时

1. 在 `manifest.json` 中登记（whl 目录名为 key，给出 triton 大版本与 SDK 小版本）；
2. 新建 `triton-<major>.md`（若该大版本尚无）与 `sdk-<minor>.md`，记录该版本实测差异与注意事项；
3. 内容要求：只写**与基础 SKILL.md 的差异**（该版本可用/不可用的 API、已验证的坑），不重复通用内容。

## 版本标识约定

- triton 大版本：取 triton wheel 版本号前两位，如 `3.6`、`3.0`；
- SDK 小版本：取 whl 版本号前三位，如 `3.7.1`、`3.7.2`；
- 若目录中缺少对应版本文件，平台会静默跳过（只有基础 skill 生效），但**建议为每个在用的版本补齐**。
