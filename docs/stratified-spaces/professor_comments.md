# Original Professor Comments

Source: pasted into the Codex thread on 2026-05-14.

The professor-commented TeX source defined these inline comment macros:

```tex
\newcommand{\xl}[1]{{\color{blue} (XL: #1)}}
\newcommand{\ads}[1]{{\color{magenta} (ADS: #1)}}
```

The pasted source contained three actual `\ads{...}` comments and no actual `\xl{...}` comment instances.

## Verbatim Comments

```tex
\ads{need more recent cites if we can find them, from 2025-2026}
```

```tex
\ads{Need a paragraph about formal testing of manifold hypothesis and main idea.}
```

```tex
\ads{need to explain here why we can more densely sample the local space around token patches.}
```

## Original Context Snippets

### Comment 1

```tex
Recent surveys and intrinsic-dimension studies give broader context for this view \citep{loaiza-ganem2024deep,levina2005maximum,facco2017estimating,pope2021intrinsic,tempczyk2022lidl,tempczyk2026benchmarks}. \ads{need more recent cites if we can find them, from 2025-2026}
```

### Comment 2

```tex
\ads{Need a paragraph about formal testing of manifold hypothesis and main idea.}
```

This appeared between the paragraph introducing the `robinson2025token` local fiber-bundle test and the paragraph beginning:

```tex
Vision gives a natural reason to revisit the question.
```

### Comment 3

```tex
This makes the local geometry easier to sample and easier to interpret. \ads{need to explain here why we can more densely sample the local space around token patches.} We can ask where in an image the local dimension changes, where the fiber-bundle slope rule is violated, and whether nearby feature-space tokens correspond to raw patches that are easy or hard to reconstruct with a sparse local dictionary.
```
