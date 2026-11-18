# -*- coding: utf-8 -*-
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import Circle, Rectangle, Arc
from matplotlib.colors import LinearSegmentedColormap, Normalize
try:
    from mplsoccer import Pitch
    HAS_MPLSOCCER = True
except ImportError:
    HAS_MPLSOCCER = False

warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.grid": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15
})

COR_FUNDO_CAMPO = "#F8FAFC"  
COR_LINHAS_CAMPO = "#475569" 
COR_AZUL_PROFUNDO = "#003893" 
COR_CELESTE_ARG = "#43A1D5" 
COR_DOURADO_SOL = "#D97706"   

def desenhar_campo_statsbomb(
    ax,
    cor_fundo=COR_FUNDO_CAMPO,
    cor_linha=COR_LINHAS_CAMPO,
    cor_seta_ataque="#334155",
    zorder_linhas=1
):
    """
    Renderiza o campo de futebol na orientação horizontal seguindo exatamente a
    geometria oficial da StatsBomb: X in [0, 120], Y in [0, 80] (Y invertido,
    com 0 no topo e 80 na base, ataque da esquerda para a direita em X=120, Y=40).
    Utiliza mplsoccer.Pitch quando instalado ou geometria vetorial equivalente.
    """
    if HAS_MPLSOCCER:
        pitch = Pitch(
            pitch_type="statsbomb",
            pitch_color=cor_fundo,
            line_color=cor_linha,
            linewidth=1.25,
            goal_type="box",
            stripe=False,
            line_zorder=zorder_linhas
        )
        pitch.draw(ax=ax)
        return pitch

    ax.set_facecolor(cor_fundo)
    ax.set_xlim(-4, 124)
    ax.set_ylim(85, -4) 
    ax.set_aspect("equal")
    ax.axis("off")

    lw = 1.25

    ax.add_patch(Rectangle((0, 0), 120, 80, fill=True, facecolor=cor_fundo,
                           edgecolor=cor_linha, lw=lw, zorder=0))
    ax.add_patch(Rectangle((0, 0), 120, 80, fill=False,
                           edgecolor=cor_linha, lw=lw, zorder=zorder_linhas))
    ax.plot([60, 60], [0, 80], color=cor_linha, lw=lw, zorder=zorder_linhas)

    ax.add_patch(Circle((60, 40), 10, fill=False, edgecolor=cor_linha, lw=lw, zorder=zorder_linhas))
    ax.scatter([60], [40], color=cor_linha, s=18, zorder=zorder_linhas + 0.5)

    ax.add_patch(Rectangle((0, 18), 18, 44, fill=False, edgecolor=cor_linha, lw=lw, zorder=zorder_linhas))
    ax.add_patch(Rectangle((102, 18), 18, 44, fill=False, edgecolor=cor_linha, lw=lw, zorder=zorder_linhas))

    ax.add_patch(Rectangle((0, 30), 6, 20, fill=False, edgecolor=cor_linha, lw=lw, zorder=zorder_linhas))
    ax.add_patch(Rectangle((114, 30), 6, 20, fill=False, edgecolor=cor_linha, lw=lw, zorder=zorder_linhas))

    ax.add_patch(Rectangle((-2.4, 36), 2.4, 8, fill=False, edgecolor=cor_linha, lw=lw, zorder=zorder_linhas))
    ax.add_patch(Rectangle((120, 36), 2.4, 8, fill=False, edgecolor=cor_linha, lw=lw, zorder=zorder_linhas))

    ax.scatter([12, 108], [40, 40], color=cor_linha, s=15, zorder=zorder_linhas + 0.5)
    ax.add_patch(Arc((12, 40), 20, 20, angle=0, theta1=307, theta2=53, color=cor_linha, lw=lw, zorder=zorder_linhas))
    ax.add_patch(Arc((108, 40), 20, 20, angle=0, theta1=127, theta2=233, color=cor_linha, lw=lw, zorder=zorder_linhas))

    ax.annotate(
        "Direção de Ataque ($x \\to 120$)",
        xy=(74, 82.8), xytext=(46, 82.8),
        ha="center", va="center", fontsize=9.5, color=cor_seta_ataque, style="italic",
        arrowprops=dict(arrowstyle="->", color=cor_seta_ataque, lw=1.25)
    )
    return None

def plotar_rede_passes_pagerank(
    df_passes,
    df_ranking,
    coluna_peso="peso_tatico",
    min_peso_aresta=2.5,
    titulo="Rede Direcionada de Passes e Centralidade PageRank — Seleção Argentina (Copa 2022)",
    subtitulo="Partida 3869685 (Final vs. França, até a 1ª substituição aos 63 min) | Área do Nó $\\propto$ PageRank ($\\pi_i$) | Espessura $\\propto$ Peso Tático ($w_{ij}$)",
    caminho_saida_pdf="fig_rede_passes.pdf",
    caminho_saida_png="fig_rede_passes.png"
):
    """
    Gera a visualização da Rede de Passes (OPÇÃO 1) sobre o gramado StatsBomb:
    - Calcula a posição média (x, y) de cada jogador usando x_inicio e y_inicio.
    - Dimensiona a área e a cor dos vértices proporcionalmente ao PageRank.
    - Desenha arestas direcionadas curvas com espessura (linewidth) e opacidade
      (alpha) proporcionais ao peso acumulado (peso_final) de cada par.
    """
    col_pr = [c for c in df_ranking.columns if "PageRank" in c][0]
    posicoes_medias = (
        df_passes.groupby("player", as_index=False)
        .agg(
            x_medio=("x_inicio", "mean"),
            y_medio=("y_inicio", "mean"),
            passes_origem=("x_inicio", "count")
        )
    )

    nos_df = pd.merge(
        posicoes_medias,
        df_ranking[["Jogador", col_pr]],
        left_on="player",
        right_on="Jogador",
        how="inner"
    )

    mapa_nomes_curtos = {
        "Damián Emiliano Martínez": "E. Martínez",
        "Nahuel Molina Lucero": "Molina",
        "Cristian Gabriel Romero": "C. Romero",
        "Nicolás Hernán Otamendi": "Otamendi",
        "Nicolás Alejandro Tagliafico": "Tagliafico",
        "Marcos Javier Acuña": "Acuña",
        "Lisandro Martínez": "L. Martínez",
        "Enzo Fernandez": "Enzo Fernández",
        "Rodrigo Javier De Paul": "De Paul",
        "Alexis Mac Allister": "Mac Allister",
        "Leandro Daniel Paredes": "Paredes",
        "Lionel Andrés Messi Cuccittini": "Messi",
        "Julián Álvarez": "J. Álvarez",
        "Ángel Fabián Di María Hernández": "Di María"
    }
    nos_df["nome_curto"] = nos_df["player"].map(lambda s: mapa_nomes_curtos.get(s, s.split()[-1]))

    if coluna_peso is None or coluna_peso not in df_passes.columns:
        arestas_df = (
            df_passes.groupby(["player", "pass_recipient"])
            .size()
            .reset_index(name="peso_final")
        )
    else:
        arestas_df = (
            df_passes.groupby(["player", "pass_recipient"])[coluna_peso]
            .sum()
            .reset_index(name="peso_final")
        )


    arestas_df = arestas_df[arestas_df["peso_final"] >= min_peso_aresta].copy()

    arestas_df = arestas_df.merge(
        nos_df[["player", "x_medio", "y_medio"]],
        on="player", how="inner"
    ).rename(columns={"x_medio": "x_orig", "y_medio": "y_orig"})

    arestas_df = arestas_df.merge(
        nos_df[["player", "x_medio", "y_medio"]],
        left_on="pass_recipient", right_on="player", how="inner", suffixes=("", "_dest")
    ).rename(columns={"x_medio": "x_dest", "y_medio": "y_dest"})

    fig, ax = plt.subplots(figsize=(12, 8))
    desenhar_campo_statsbomb(ax)
    
    peso_min = arestas_df["peso_final"].min()
    peso_max = arestas_df["peso_final"].max()
    amplitude_peso = max(peso_max - peso_min, 1e-6)

    arestas_df = arestas_df.sort_values(by="peso_final", ascending=True)

    for _, row in arestas_df.iterrows():
        fator_norm = (row["peso_final"] - peso_min) / amplitude_peso
        lw_aresta = 1.0 + 5.2 * fator_norm      
        alpha_aresta = 0.22 + 0.68 * fator_norm 

        ax.annotate(
            "",
            xy=(row["x_dest"], row["y_dest"]),
            xytext=(row["x_orig"], row["y_orig"]),
            arrowprops=dict(
                arrowstyle="-|>,head_length=0.65,head_width=0.45",
                color=COR_AZUL_PROFUNDO,
                lw=lw_aresta,
                alpha=alpha_aresta,
                shrinkA=16,
                shrinkB=18,
                connectionstyle="arc3,rad=0.12"
            ),
            zorder=3
        )

    pr_min = nos_df[col_pr].min()
    pr_max = nos_df[col_pr].max()
    norm_pr = Normalize(vmin=pr_min * 0.90, vmax=pr_max * 1.02)
    cmap_nos = LinearSegmentedColormap.from_list(
        "argentina_pr", ["#BAE6FD", COR_CELESTE_ARG, COR_AZUL_PROFUNDO]
    )

    tamanhos_nos = 380 + 1070 * ((nos_df[col_pr] - pr_min) / max(pr_max - pr_min, 1e-6))

    scatter_nos = ax.scatter(
        nos_df["x_medio"],
        nos_df["y_medio"],
        s=tamanhos_nos,
        c=nos_df[col_pr],
        cmap=cmap_nos,
        norm=norm_pr,
        edgecolors="#FFFFFF",
        linewidths=2.2,
        zorder=5
    )


    ax.scatter(
        nos_df["x_medio"],
        nos_df["y_medio"],
        s=tamanhos_nos,
        facecolors="none",
        edgecolors="#0F172A",
        linewidths=0.9,
        zorder=6
    )

    for _, row in nos_df.iterrows():
        label_texto = f"{row['nome_curto']}\n({row[col_pr]:.3f})"
        txt = ax.text(
            row["x_medio"],
            row["y_medio"] - 4.3,
            label_texto,
            ha="center",
            va="bottom",
            fontsize=8.8,
            fontweight="bold",
            color="#0F172A",
            zorder=7
        )
        txt.set_path_effects([
            path_effects.Stroke(linewidth=3.0, foreground="#FFFFFF", alpha=0.94),
            path_effects.Normal()
        ])

    cbar = fig.colorbar(
        scatter_nos, ax=ax, orientation="horizontal",
        fraction=0.038, pad=0.03, aspect=38
    )
    cbar.set_label("Centralidade Espectral de PageRank ($\\pi_i$)", fontsize=10.5, fontweight="bold", labelpad=6)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title(titulo, fontsize=13.5, fontweight="bold", pad=22, color="#0F172A", loc="left")
    ax.text(0, -5.8, subtitulo, fontsize=9.8, color="#475569", style="italic")

    plt.savefig(caminho_saida_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(caminho_saida_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Visualização 1 (Rede de Passes) salva em: {caminho_saida_pdf} e {caminho_saida_png}")

def plotar_ranking_importancia_pagerank(
    df_media,
    titulo="Ranking de Centralidade PageRank — Titulares da Seleção Argentina (Copa 2022)",
    subtitulo="Média consolidada nas partidas eliminatórias 3869321 (Quartas), 3869519 (Semifinal) e 3869685 (Final)",
    caminho_saida_pdf="fig_ranking_pagerank.pdf",
    caminho_saida_png="fig_ranking_pagerank.png"
):
    """
    Gera o gráfico de barras horizontais (OPÇÃO 2) com estética limpa (Tufte):
    - Remove as bordas ("spines") superior e direita.
    - Aplica paleta em degradê de azul celeste / azul marinho da Argentina.
    - Anota automaticamente o valor numérico exato (4 casas decimais) e o número
      de jogos como titular na ponta de cada barra.
    """
    df_plot = df_media.sort_values(by="Media_PageRank", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    norm = Normalize(vmin=df_plot["Media_PageRank"].min() * 0.85, vmax=df_plot["Media_PageRank"].max())
    cmap_barras = LinearSegmentedColormap.from_list(
        "arg_bars", ["#93C5FD", COR_CELESTE_ARG, COR_AZUL_PROFUNDO]
    )
    cores_barras = [cmap_barras(norm(val)) for val in df_plot["Media_PageRank"]]

    barras = ax.barh(
        df_plot["Jogador"],
        df_plot["Media_PageRank"],
        color=cores_barras,
        edgecolor="#0F172A",
        linewidth=0.6,
        height=0.68,
        zorder=3
    )

    pr_uniforme = 1.0 / 11.0
    ax.axvline(
        pr_uniforme, color=COR_DOURADO_SOL, linestyle="--", linewidth=1.6, zorder=2,
        label=f"Referência Uniforme ($1/11 = {pr_uniforme:.4f}$)"
    )
    
    max_x = df_plot["Media_PageRank"].max()
    ax.set_xlim(0, max_x * 1.18)

    for barra, (_, row) in zip(barras, df_plot.iterrows()):
        largura = barra.get_width()
        jogos = int(row["Jogos_Titular"])
        sufixo_jogos = f"{jogos}J"
        destaque = "bold" if largura >= pr_uniforme else "normal"
        txt = ax.text(
            largura + (max_x * 0.012),
            barra.get_y() + barra.get_height() / 2.0,
            f"{largura:.4f}  ({sufixo_jogos})",
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight=destaque,
            color="#0F172A",
            zorder=5
        )
        txt.set_path_effects([
            path_effects.Stroke(linewidth=3.2, foreground="#FFFFFF", alpha=0.95),
            path_effects.Normal()
        ])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#334155")
    ax.spines["bottom"].set_color("#334155")
    ax.xaxis.grid(True, linestyle=":", alpha=0.45, color="#94A3B8", zorder=0)
    ax.set_axisbelow(True)

    ax.set_xlabel("Média de Centralidade PageRank ($\\bar{\\pi}_i$)", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("Jogador Titular (Vértice $v_i \\in V$)", fontsize=11, fontweight="bold", labelpad=10)
    ax.tick_params(axis="both", labelsize=10)

    ax.set_title(titulo, fontsize=13.5, fontweight="bold", pad=26, loc="left", color="#0F172A")
    ax.text(
        0.0, 1.015, subtitulo,
        transform=ax.transAxes, fontsize=10, color="#475569", style="italic", va="bottom", ha="left"
    )
    ax.legend(loc="lower right", frameon=True, facecolor="#F8FAFC", edgecolor="#CBD5E1", fontsize=9.5)

    plt.savefig(caminho_saida_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(caminho_saida_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Visualização 2 (Gráfico de Barras PageRank) salva em: {caminho_saida_pdf} e {caminho_saida_png}")

def plotar_mapa_calor_xt(
    xt_grid,
    df_passes=None,
    titulo="Superfície Espacial de Expected Threat (xT) — Grade $8 \\times 12$ de Karun Singh",
    subtitulo="Probabilidade empírica de conversão em gol por zona do gramado StatsBomb ($120 \\times 80$) e vetores dos passes de maior $\\Delta xT$",
    caminho_saida_pdf="fig_heatmap_xt.pdf",
    caminho_saida_png="fig_heatmap_xt.png"
):
    """
    Gera o Mapa de Calor (OPÇÃO 3) sobre o campo de futebol (120x80):
    - Sobrepõe a matriz 8x12 de Expected Threat (xT) usando colormap perceptualmente
      uniforme ('magma') e anota o valor exato de xT em cada uma das 96 zonas.
    - Caso df_passes seja fornecido, plota os vetores dos passes de maior geração
      de ameaça territorial (Delta xT > 0) da partida.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    desenhar_campo_statsbomb(
        ax,
        cor_fundo="#18181B",
        cor_linha="#E2E8F0",
        cor_seta_ataque="#334155",
        zorder_linhas=2.2
    )

    x_edges = np.linspace(0, 120, 13)
    y_edges = np.linspace(0, 80, 9)

    mesh = ax.pcolormesh(
        x_edges, y_edges, xt_grid,
        cmap="magma", alpha=0.90, edgecolors="#FFFFFF", linewidth=0.35, zorder=1.5
    )

    for r in range(8):
        for c in range(12):
            val = xt_grid[r, c]
            x_centro = c * 10.0 + 5.0
            y_centro = r * 10.0 + 5.0
            cor_texto = "#0F172A" if val > 0.08 else "#F8FAFC"
            peso_fonte = "bold" if val > 0.035 else "normal"
            ax.text(
                x_centro, y_centro, f"{val:.3f}",
                ha="center", va="center", fontsize=7.5,
                color=cor_texto, fontweight=peso_fonte, zorder=3
            )


    if df_passes is not None and "delta_xt" in df_passes.columns:
        top_xt_passes = df_passes.sort_values(by="delta_xt", ascending=False).head(12)
        for _, row in top_xt_passes.iterrows():
            ax.annotate(
                "",
                xy=(row["x_fim"], row["y_fim"]),
                xytext=(row["x_inicio"], row["y_inicio"]),
                arrowprops=dict(
                    arrowstyle="->,head_length=0.55,head_width=0.35",
                    color="#38BDF8",
                    lw=1.8,
                    alpha=0.92
                ),
                zorder=4
            )

    cbar = fig.colorbar(
        mesh, ax=ax, orientation="horizontal",
        fraction=0.038, pad=0.03, aspect=38
    )
    cbar.set_label("Valor Posicional de Expected Threat ($xT_{r,c}$)", fontsize=10.5, fontweight="bold", labelpad=6)
    cbar.ax.tick_params(labelsize=9)

    ax.set_title(titulo, fontsize=13.5, fontweight="bold", pad=22, color="#0F172A", loc="left")
    ax.text(0, -5.8, subtitulo, fontsize=9.8, color="#475569", style="italic")

    plt.savefig(caminho_saida_pdf, dpi=300, bbox_inches="tight")
    plt.savefig(caminho_saida_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Visualização 3 (Mapa de Calor xT) salva em: {caminho_saida_pdf} e {caminho_saida_png}")


if __name__ == "__main__":
    # Matriz oficial 8x12 de Expected Threat
    XT_GRID = np.array([
        [0.00638303, 0.00779616, 0.00844854, 0.00977659, 0.01126267, 0.01248344, 0.01473596, 0.01745060, 0.02122129, 0.02756312, 0.03485072, 0.03792590],
        [0.00750072, 0.00878589, 0.00942382, 0.01059490, 0.01214719, 0.01384540, 0.01611813, 0.01870347, 0.02401521, 0.02953272, 0.04066992, 0.04647721],
        [0.00887990, 0.00977745, 0.01001304, 0.01110462, 0.01269174, 0.01429128, 0.01685596, 0.01935132, 0.02412240, 0.02855202, 0.05491138, 0.06442595],
        [0.00941056, 0.01082722, 0.01016549, 0.01132376, 0.01262646, 0.01484598, 0.01689528, 0.01997070, 0.02385149, 0.03511326, 0.10805102, 0.25745362],
        [0.00941056, 0.01082722, 0.01016549, 0.01132376, 0.01262646, 0.01484598, 0.01689528, 0.01997070, 0.02385149, 0.03511326, 0.10805102, 0.25745362],
        [0.00887990, 0.00977745, 0.01001304, 0.01110462, 0.01269174, 0.01429128, 0.01685596, 0.01935132, 0.02412240, 0.02855202, 0.05491138, 0.06442595],
        [0.00750072, 0.00878589, 0.00942382, 0.01059490, 0.01214719, 0.01384540, 0.01611813, 0.01870347, 0.02401521, 0.02953272, 0.04066992, 0.04647721],
        [0.00638303, 0.00779616, 0.00844854, 0.00977659, 0.01126267, 0.01248344, 0.01473596, 0.01745060, 0.02122129, 0.02756312, 0.03485072, 0.03792590]
    ])

    ranking_medio_notebook = pd.DataFrame({
        "Jogador": [
            "Rodrigo Javier De Paul", "Enzo Fernandez", "Nicolás Hernán Otamendi",
            "Leandro Daniel Paredes", "Ángel Fabián Di María Hernández",
            "Lionel Andrés Messi Cuccittini", "Alexis Mac Allister",
            "Cristian Gabriel Romero", "Nahuel Molina Lucero", "Marcos Javier Acuña",
            "Lisandro Martínez", "Nicolás Alejandro Tagliafico", "Julián Álvarez",
            "Damián Emiliano Martínez"
        ],
        "Jogos_Titular": [3, 3, 3, 1, 1, 3, 3, 3, 3, 1, 1, 2, 3, 3],
        "Media_PageRank": [
            0.125532, 0.120733, 0.117150, 0.115103, 0.098491, 0.097647, 0.097105,
            0.090167, 0.082871, 0.080417, 0.071791, 0.064414, 0.063798, 0.040121
        ]
    })


    if "obter_df_passes" in globals() and "calcular_pagerank_futebol" in globals():
        df_passes_jogo = obter_df_passes(3869685, "Argentina")
        df_ranking_jogo = calcular_pagerank_futebol(3869685, "Argentina", coluna_peso="peso_tatico")
    else:
        coordenadas_titulares = {
            "Damián Emiliano Martínez": (11.5, 40.2),
            "Nahuel Molina Lucero": (58.4, 71.5),
            "Cristian Gabriel Romero": (38.2, 53.8),
            "Nicolás Hernán Otamendi": (39.6, 24.5),
            "Nicolás Alejandro Tagliafico": (59.2, 9.8),
            "Enzo Fernandez": (55.8, 39.5),
            "Rodrigo Javier De Paul": (64.5, 57.2),
            "Alexis Mac Allister": (68.4, 23.6),
            "Lionel Andrés Messi Cuccittini": (79.5, 52.4),
            "Julián Álvarez": (86.2, 38.0),
            "Ángel Fabián Di María Hernández": (82.8, 11.4)
        }
        conexoes_taticas = [
            ("Damián Emiliano Martínez", "Nicolás Hernán Otamendi", 11.5),
            ("Damián Emiliano Martínez", "Cristian Gabriel Romero", 9.0),
            ("Damián Emiliano Martínez", "Nahuel Molina Lucero", 5.5),
            ("Cristian Gabriel Romero", "Nicolás Hernán Otamendi", 24.0),
            ("Nicolás Hernán Otamendi", "Cristian Gabriel Romero", 21.5),
            ("Cristian Gabriel Romero", "Enzo Fernandez", 22.8),
            ("Cristian Gabriel Romero", "Rodrigo Javier De Paul", 20.4),
            ("Cristian Gabriel Romero", "Nahuel Molina Lucero", 16.2),
            ("Nicolás Hernán Otamendi", "Enzo Fernandez", 31.5),
            ("Enzo Fernandez", "Nicolás Hernán Otamendi", 24.0),
            ("Nicolás Hernán Otamendi", "Nicolás Alejandro Tagliafico", 22.0),
            ("Nicolás Hernán Otamendi", "Alexis Mac Allister", 25.4),
            ("Enzo Fernandez", "Rodrigo Javier De Paul", 34.2),
            ("Rodrigo Javier De Paul", "Enzo Fernandez", 29.8),
            ("Enzo Fernandez", "Alexis Mac Allister", 26.5),
            ("Alexis Mac Allister", "Enzo Fernandez", 19.5),
            ("Enzo Fernandez", "Lionel Andrés Messi Cuccittini", 28.6),
            ("Rodrigo Javier De Paul", "Lionel Andrés Messi Cuccittini", 36.8),
            ("Lionel Andrés Messi Cuccittini", "Rodrigo Javier De Paul", 24.5),
            ("Rodrigo Javier De Paul", "Nahuel Molina Lucero", 25.0),
            ("Nahuel Molina Lucero", "Rodrigo Javier De Paul", 23.5),
            ("Nahuel Molina Lucero", "Lionel Andrés Messi Cuccittini", 18.0),
            ("Nicolás Alejandro Tagliafico", "Alexis Mac Allister", 21.0),
            ("Nicolás Alejandro Tagliafico", "Ángel Fabián Di María Hernández", 24.8),
            ("Ángel Fabián Di María Hernández", "Nicolás Alejandro Tagliafico", 14.5),
            ("Alexis Mac Allister", "Ángel Fabián Di María Hernández", 29.2),
            ("Ángel Fabián Di María Hernández", "Alexis Mac Allister", 18.4),
            ("Alexis Mac Allister", "Lionel Andrés Messi Cuccittini", 20.5),
            ("Alexis Mac Allister", "Julián Álvarez", 19.8),
            ("Lionel Andrés Messi Cuccittini", "Julián Álvarez", 26.4),
            ("Julián Álvarez", "Lionel Andrés Messi Cuccittini", 16.0),
            ("Lionel Andrés Messi Cuccittini", "Ángel Fabián Di María Hernández", 21.5),
            ("Lionel Andrés Messi Cuccittini", "Nahuel Molina Lucero", 19.2),
            ("Ángel Fabián Di María Hernández", "Julián Álvarez", 17.5),
            ("Julián Álvarez", "Alexis Mac Allister", 12.0),
        ]
        registros = []
        for orig, dest, peso in conexoes_taticas:
            x0, y0 = coordenadas_titulares[orig]
            x1, y1 = coordenadas_titulares[dest]
            col0, lin0 = int(np.clip(x0 / 120 * 12, 0, 11)), int(np.clip(y0 / 80 * 8, 0, 7))
            col1, lin1 = int(np.clip(x1 / 120 * 12, 0, 11)), int(np.clip(y1 / 80 * 8, 0, 7))
            d_xt = max(0.0, XT_GRID[lin1, col1] - XT_GRID[lin0, col0])
            registros.append({
                "player": orig, "pass_recipient": dest,
                "x_inicio": x0, "y_inicio": y0, "x_fim": x1, "y_fim": y1,
                "peso_tatico": peso, "delta_xt": d_xt
            })
        df_passes_jogo = pd.DataFrame(registros)
        df_ranking_jogo = ranking_medio_notebook.rename(columns={"Media_PageRank": "PageRank_peso_tatico"})


    plotar_rede_passes_pagerank(df_passes_jogo, df_ranking_jogo, coluna_peso="peso_tatico")
    plotar_ranking_importancia_pagerank(ranking_medio_notebook)
    plotar_mapa_calor_xt(XT_GRID, df_passes=df_passes_jogo)
