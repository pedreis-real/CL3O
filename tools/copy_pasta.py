import matplotlib.pyplot as plt
import numpy as np

def plot_table12():
    labels = ['XC', 'ZC', 'A', 'IXX', 'IZZ', 'IXZ', 'I1', 'I2', 'θP', 'J', 'XS', 'ZS']
    
    # Simétrico
    femap_sim = np.abs([134.27, -0.14, 1318.56, 4244060, 9178561, -9297, 9178571, 4244050, -89.94, 771313, 77.07, 0.25])
    cl3o_sim = np.abs([136.72, 0.16, 1411.97, 4256838, 10440413, -11400, 10440425, 4256826, -89.94, 829828, 91.67, -0.76])
    
    # Assimétrico
    femap_assim = np.abs([116.24, 13.53, 1782.43, 354030, 11053338, 211034, 11057499, 349869, 91.13, 1022279, 85.69, 5.86])
    cl3o_assim = np.abs([121.34, 13.10, 1941.51, 415994, 13698836, 116288, 13699854, 414976, 89.50, 1155441, 91.12, 20.25])
    
    x = np.arange(len(labels))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(x - 1.5*width, femap_sim, width, label='Femap (Simétrico)', color='skyblue')
    ax.bar(x - 0.5*width, cl3o_sim, width, label='CL3O (Simétrico)', color='steelblue')
    ax.bar(x + 0.5*width, femap_assim, width, label='Femap (Assimétrico)', color='lightgreen')
    ax.bar(x + 1.5*width, cl3o_assim, width, label='CL3O (Assimétrico)', color='forestgreen')
    
    ax.set_ylabel('Valores (abs)')
    ax.set_title('Tabela 12: Propriedades Geométricas - Femap vs CL3O')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_yscale('symlog', linthresh=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('grafico_tabela_12.pdf', dpi=300)
    plt.close()

def plot_table13():
    labels = ['2D RZ', '2D MY', '3D RX', '3D RY', '3D RZ', '3D MX', '3D MY', '3D MZ']
    femap = np.abs([-1000.00, 3.00e6, -10.00, 0.00, -1000.00, 3.24e6, 1.43e5, 3.24e4])
    cl3o = np.abs([-1000.00, 2.97e6, -10.00, 0.00, -1000.00, 3.24e6, 3.37e5, 3.24e4])
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, femap, width, label='Femap', color='coral')
    ax.bar(x + width/2, cl3o, width, label='CL3O', color='firebrick')
    
    ax.set_ylabel('Esforços Reativos (abs)')
    ax.set_title('Tabela 13: Comparativo dos Esforços Reativos no Engaste')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_yscale('symlog', linthresh=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('grafico_tabela_13.pdf', dpi=300)
    plt.close()

def plot_table14():
    labels = ['2D uZ', '2D θX', '3D uX', '3D uY', '3D uZ', '3D θX', '3D θY', '3D θZ']
    femap = np.abs([130.80, 3.68, -5.17, 13.32, 166.30, 4.30, -0.90, 0.05])
    cl3o = np.abs([184.30, 5.37, -9.27, -0.70, 237.50, 6.41, -1.41, 0.32])
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, femap, width, label='Femap', color='mediumpurple')
    ax.bar(x + width/2, cl3o, width, label='CL3O', color='rebeccapurple')
    
    ax.set_ylabel('Deslocamentos (abs)')
    ax.set_title('Tabela 14: Comparativo dos Deslocamentos na Extremidade Livre')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_yscale('symlog', linthresh=0.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('grafico_tabela_14.pdf', dpi=300)
    plt.close()

def plot_table15():
    labels = ['2D uZ', '2D θX', '3D uX', '3D uY', '3D uZ', '3D θX', '3D θY', '3D θZ']
    femap = np.abs([130.80, 3.68, -5.17, 13.32, 166.30, 4.30, -0.90, 0.05])
    cl3o = np.abs([131.00, 3.67, -0.97, -0.70, 168.40, 6.41, -1.41, 0.32])
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, femap, width, label='Femap', color='gold')
    ax.bar(x + width/2, cl3o, width, label='CL3O', color='darkgoldenrod')
    
    ax.set_ylabel('Deslocamentos (abs)')
    ax.set_title('Tabela 15: Comparativo dos Deslocamentos na Extremidade Livre')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_yscale('symlog', linthresh=0.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('grafico_tabela_15.pdf', dpi=300)
    plt.close()

plot_table12()
plot_table13()
plot_table14()
plot_table15()
print("Plots generated successfully.")