---
name: latex-doc-agent
description: 用 LaTeX 编写并编译"详细解析类"技术 PDF（中文为主）。输入一份源 PDF/讲义，产出概念详解 + 生动类比 + 全注释代码的精讲文档。适用于"把讲义/论文/教程改写成更易懂的 PDF"、以及任何需要排版精美中文 PDF 的任务。
---

# latex-doc-agent：LaTeX 详细解析 PDF

把一份技术讲义改写成"概念讲透、类比生动、代码注释齐全"的中文 LaTeX PDF。
核心不是排版技巧，而是**写作纪律**：先读透源材料，再定结构，再逐章写，最后编译验证。

## 0. 前置：先读透源 PDF（不要跳读）
用 pypdf 按页提取全文，超过约 10 页必须分段读全；记录每页的
「概念 / 数据 / 代码 / 图注」，否则后续"解析"会变成自信的编造。
~~~python
python -c "from pypdf import PdfReader; [print(p.extract_text()) for p in PdfReader('src.pdf').pages]"
~~~

## 1. 定目标结构（先于写作）
- 结构来自：源讲义自身脉络 + 用户要求 + 常见教学结构（导读→概念→逐版拆解→理论→总结→附录）。
- 先把「章节 → 覆盖的源材料」映射成表，暴露两个风险：无源支撑的章节（标注待确认，不许编造）、
  无处安放的源材料（问是否进附录）。
- 给出**大纲表格**让用户确认后再逐章写。

## 2. 写作三件套（本 Skill 的价值所在）
对每个核心概念/每版代码，都配齐：
1. **概念解析**：是什么 → 为什么 → 怎么做 → 常见坑；
2. **生动类比**：用「\analogy{...}」盒子；要求"贴近生活、只对应当前概念、不误导"，
   同义类比尽量不重复使用；
3. **全注释代码**：「listings」代码块按"三层注释"标准（见 README 的规范）：
   段前块注释（做什么+类比）、行尾注释（为什么）、代码后的"为什么慢/快"列表（瓶颈+动机）。

## 3. 工程纪律
- 「main.tex」单入口 + 「chapters/」按章拆分 + 「figures/」图源（TikZ 源码化，不手绘位图）；
- 中文必须「xelatex」+「ctexart」（本机 MiKTeX 已验证）；**不要用 pdflatex 跑中文**；
- 本机「latexmk」缺 perl → 直接用「xelatex」连跑 2-3 遍（TOC/交叉引用需要多趟）；
- 先让"空壳工程"编译通过，再逐章填充，每填一章编译一次，尽早暴露错误。

## 4. 编译与验证
~~~powershell
powershell -ExecutionPolicy Bypass -File build.ps1   # xelatex 三遍 + 日志检查
~~~
- 编译后必须查「main.log」：「Overfull/Underfull hbox」（表格/图过宽）、
  「Undefined reference/control sequence」（\label 缺失或笔误）——这些**不报错但会在 PDF 里静默出错**；
- 用 pypdf 抽一段 PDF 文本核对中文/代码是否渲染正确；
- 交付：「main.pdf」+ 全部「.tex」源码 + 本 Skill 说明。

## 5. 注意事项
- 不编造源 PDF 里没有的数据/结论；原讲义有 OCR 噪声时，在附录给"勘误表"而不是静默改正；
- 全注释代码以"能编译、能看懂、注释>代码"为准，宁可注释多不可少（用户明确要求）；
- 现代 GPU 内容（shuffle、cooperative groups）只作附录对照，不混入原讲义主体脉络。
