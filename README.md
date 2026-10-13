# video-ia-anatomia
animação ref ia instagram

## v2: `output/gluteo_celulite_v2.mp4` (versão atual)

1080×1920, 30 fps, 15 s, com a trilha do vídeo de referência.

| Tempo | Plano | Conteúdo |
|---|---|---|
| 0,0–2,0 s | Filmagem real do ANTES, de costas | Push-in estabilizado por rastreamento; anéis finos acompanham 3 depressões reais; foco direcionado |
| 2,0–4,75 s | Filmagem real do ANTES, arco lateral | Movimento real da câmera mostrando a celulite lateral, depois mergulho na pele (zoom com blur radial) |
| 4,75–6,9 s | Imagem de IA enviada | Saindo da pele: revelação das camadas (pele, gordura, septos, músculo, vasos, osso) com passada de luz. Na pele, depressões puxadas pelas traves |
| 6,9–9,35 s | Close nos septos | Traves fibrosas tensas ligam a gordura à camada septal e puxam a gordura para baixo. A agulha entra e rompe trave por trave, e as fibras recuam e se soltam |
| 9,5–11,2 s | Bioestimulador | Ao recuar, a agulha deposita o gel ao longo do túnel e fibrilas de colágeno crescem a partir dos depósitos |
| 11,2–12,05 s | Pele | A câmera sobe para a pele, as depressões se desfazem e há um mergulho de volta |
| 12,05–15,0 s | Filmagem real do DEPOIS | Abre do close para o plano inteiro |

**A filmagem do antes não foi alterada**: sem retoque, sem realce e com as marcações de caneta. Foram adicionados só enquadramento/movimento digital, anéis de destaque, uma gradação global igual para o vídeo todo (contraste suave, halation, vinheta e grão) e as transições.

Código: `pipeline/render_v2.py`. Os tempos ficam em `SHOT_A / SHOT_B / SHOT_AF / AI_CAM / T_IN / T_OUT`, as traves em `CORDS` e as depressões da pele de IA em `DIMPLES`.

```bash
mkdir -p work/bf work/af
ffmpeg -i "WhatsApp Video 2026-10-13 at 09.54.17.mp4" -t 10.5 -q:v 2 work/bf/%04d.jpg
ffmpeg -ss 22 -i "WhatsApp Video 2026-10-13 at 09.54.15.mp4" -t 6 -q:v 2 work/af/%04d.jpg
python3 pipeline/render_v2.py work assets/ai/camadas_gluteo.jpg out_noaudio.mp4
ffmpeg -i out_noaudio.mp4 -i "WhatsApp Video 2026-10-13 at 09.49.07.mp4" -map 0:v -map 1:a -c:v copy -c:a aac -shortest final.mp4
```

---

## v1 (anterior): `output/gluteo_celulite_remake_v1.mp4`

1080×1920, 30 fps, 15 s, com a trilha do vídeo de referência. Segue a mesma estrutura e o mesmo ritmo do vídeo original, agora com o novo caso.

| Tempo | Beat | O que acontece |
|---|---|---|
| 0,0–1,0 s | Problema visível | Foto real do ANTES com push-in lento |
| 1,0–2,7 s | Leitura | Linha fina traçando a silhueta e anéis finos marcando as depressões reais |
| 2,4–4,3 s | Ênfase | Aproximação nos pontos principais. O relevo da própria foto é aprofundado: sombras de contato e bordas |
| 4,3–6,5 s | Ver através da pele | A pele fica luminosa e translúcida, depois a gordura e os septos aparecem, a partir das depressões |
| 6,5–6,8 s | Pontos de tração | Luz quente localizada e septos convergindo para cada depressão |
| 6,8–10,3 s | Procedimento | Cânula entra lateralmente e faz leque sob cada ponto. O septo solta, a depressão suaviza aos poucos. Fibrilas de bioestimulação bem discretas |
| 10,3–11,4 s | Volta à pele | Tecidos somem, textura real retorna e a câmera abre |
| 11,4–12,9 s | Morph | Warp TPS por landmarks anatômicos entre as duas fotos reais |
| 12,9–15,0 s | Resultado | Foto real do DEPOIS com anéis nos mesmos pontos |

### Princípio: a fotografia comanda
- Nenhum corpo é gerado. Todas as camadas são renderizadas no espaço de pixels da própria foto do ANTES e iluminadas com a luz de baixa frequência da própria foto. Por isso posição, enquadramento, proporção e formato do glúteo nunca mudam.
- As depressões realçadas e tratadas são as **detectadas na foto** (`pipeline/prep.py`, mapa passa-banda de luminância) e depois curadas manualmente (`pipeline/landmarks.py: TETHERS`). Nenhuma celulite nova é criada. O realce tem teto suave (`tanh`) para não virar cratera, e os sulcos infraglúteos, a fenda e a silhueta ficam fora dele.
- A transição ANTES → DEPOIS usa só escala + translação (sem rotação) e um warp thin-plate-spline guiado por landmarks de cintura, calcinha, quadril, sulcos, fenda e coxas. Joelhos e panturrilhas ficam fora do warp, porque as pernas estão mais afastadas na foto do depois e deformá-las faria o corpo "derreter". Ali as fotos simplesmente fazem dissolve.

### Intervenções na imagem (transparência)
- **Traços de caneta da marcação** foram removidos da foto do ANTES por inpainting, só em traços finos dentro da pele (máscara em `prep.py`).
- **Balanço de branco do DEPOIS**: deslocamento global de média em LAB (55 %), medido na pele, para atenuar a diferença de iluminação entre as sessões. Nenhum ajuste local ou de forma.
- Gradação única no vídeo inteiro: contraste suave, vinheta, nitidez leve e grão.

### Fontes
- ANTES: `WhatsApp Video 2026-10-13 at 09.54.17`, quadro em ~3,0 s → `assets/case/before_frame.png`
- DEPOIS: `WhatsApp Video 2026-10-13 at 09.54.15`, quadro em ~26,0 s → `assets/case/after_frame.png`
- Referência de ritmo e áudio: `WhatsApp Video 2026-10-13 at 09.49.07`

A pose usada é de costas, reta. É a única que existe nos dois vídeos com alinhamento confiável: as vistas 3/4 de cada vídeo são de lados opostos.

### Reproduzir
```bash
pip install opencv-contrib-python-headless numpy
mkdir work && cp assets/case/before_frame.png work/before.png && cp assets/case/after_frame.png work/after.png
python3 pipeline/prep.py work
python3 pipeline/render.py work out_noaudio.mp4             # ~9 min em CPU
python3 pipeline/render.py work prev.mp4 --preview 3.8,8.0  # quadros soltos
ffmpeg -i out_noaudio.mp4 -i <referência>.mp4 -map 0:v -map 1:a -c:v copy -c:a aac -shortest final.mp4
```
Os tempos dos beats ficam em `T_SCAN / T_ZOOM / T_REVEAL / PROC / T_BACK / T_MORPH`, a câmera em `CAM` e a ordem de tratamento em `_build_schedule` (`render.py`).

### Limites conhecidos
- As fontes são vídeos de WhatsApp (576×1024 e 576×768). O close chega a ~3,5× de ampliação, então a nitidez máxima é limitada pela origem. Exportar os vídeos originais do celular melhora tudo direto.
- A camada de gordura e septos é procedural, construída pixel a pixel sobre a foto. Isso garante alinhamento perfeito, mas não tem o nível de detalhe de um modelo generativo de vídeo. Se quiser um acabamento a mais, os quadros desta versão podem servir de keyframes travados para uma ferramenta image-to-video (Kling, Runway, Veo), sempre com o quadro real como primeiro e último frame.
