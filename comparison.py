import numpy as np

def compare(ppo_scores=None, sb3_scores=None, random_scores=None, model_names=None):
    """
    Confronto semplice tra modelli già trainati usando i loro scores
    
    Args:
        ppo_scores (list): Lista degli score del PPO custom
        sb3_scores (list): Lista degli score del SB3 PPO  
        random_scores (list): Lista degli score del Random Agent
        model_names (dict): Nomi custom per i modelli (opzionale)
    
    Returns:
        dict: Risultati del confronto
    """
    
    # Nomi di default
    default_names = {
        'ppo': 'PPO Custom',
        'sb3': 'SB3 PPO', 
        'random': 'Random Agent'
    }
    
    if model_names:
        default_names.update(model_names)
    
    # Raccogli i modelli disponibili
    models = {}
    
    if ppo_scores is not None:
        models['ppo'] = {
            'name': default_names['ppo'],
            'scores': ppo_scores,
            'final_avg': np.mean(ppo_scores[-50:]) if len(ppo_scores) >= 50 else np.mean(ppo_scores),
            'overall_avg': np.mean(ppo_scores),
            'max_score': np.max(ppo_scores),
            'min_score': np.min(ppo_scores),
            'std': np.std(ppo_scores),
            'episodes': len(ppo_scores)
        }
    
    if sb3_scores is not None:
        models['sb3'] = {
            'name': default_names['sb3'],
            'scores': sb3_scores,
            'final_avg': np.mean(sb3_scores[-50:]) if len(sb3_scores) >= 50 else np.mean(sb3_scores),
            'overall_avg': np.mean(sb3_scores),
            'max_score': np.max(sb3_scores),
            'min_score': np.min(sb3_scores),
            'std': np.std(sb3_scores),
            'episodes': len(sb3_scores)
        }
    
    if random_scores is not None:
        models['random'] = {
            'name': default_names['random'],
            'scores': random_scores,
            'final_avg': np.mean(random_scores[-50:]) if len(random_scores) >= 50 else np.mean(random_scores),
            'overall_avg': np.mean(random_scores),
            'max_score': np.max(random_scores),
            'min_score': np.min(random_scores),
            'std': np.std(random_scores),
            'episodes': len(random_scores)
        }
    
    print(f"\nCONFRONTO MODELLI ({len(models)} modelli)")
    print("="*80)
    
    # Header tabella
    print(f"{'Pos':<4} {'Modello':<15} {'Score Finale':<12} {'Score Medio':<12} {'Max':<8} {'Std':<8} {'Episodi':<8}")
    print("-"*80)
    
    # Ordina per performance finale
    sorted_models = sorted(models.items(), key=lambda x: x[1]['final_avg'], reverse=True)
    
    medals = ["🥇", "🥈", "🥉"]
    
    for i, (key, data) in enumerate(sorted_models):
        medal = medals[i] if i < 3 else f"{i+1}°"
        print(f"{medal:<4} {data['name']:<15} {data['final_avg']:>10.1f}  {data['overall_avg']:>10.1f}  "
              f"{data['max_score']:>6.0f}  {data['std']:>6.1f}  {data['episodes']:>6d}")
    
    # Statistiche aggiuntive se abbiamo più di un modello
    if len(models) > 1:
        print(f"\nANALISI DETTAGLIATA")
        print("-"*50)
        
        # Trova il migliore e peggiore
        best_model = sorted_models[0]
        worst_model = sorted_models[-1]
        
        print(f"Migliore: {best_model[1]['name']} ({best_model[1]['final_avg']:.1f})")
        print(f"Peggiore: {worst_model[1]['name']} ({worst_model[1]['final_avg']:.1f})")
        
        if len(models) >= 2:
            diff = best_model[1]['final_avg'] - worst_model[1]['final_avg']
            improvement = (diff / worst_model[1]['final_avg']) * 100 if worst_model[1]['final_avg'] > 0 else float('inf')
            print(f"📈 Differenza: {diff:.1f} punti (+{improvement:.0f}%)")
        
        # Confronto con Random se presente
        if 'random' in models and len(models) > 1:
            print(f"\nMIGLIORAMENTO vs RANDOM:")
            random_avg = models['random']['final_avg']
            for key, data in models.items():
                if key != 'random':
                    if random_avg > 0:
                        improvement = ((data['final_avg'] - random_avg) / random_avg) * 100
                        print(f"   {data['name']}: +{improvement:.0f}%")
                    else:
                        print(f"   {data['name']}: Score assoluto: {data['final_avg']:.1f}")
        
        # Analisi stabilità
        print(f"\nSTABILITÀ (minore std = più stabile):")
        stability_ranking = sorted(models.items(), key=lambda x: x[1]['std'])
        for i, (key, data) in enumerate(stability_ranking):
            stability_medal = "🎯" if i == 0 else "⚡" if i == 1 else "📊"
            print(f"   {stability_medal} {data['name']}: σ = {data['std']:.1f}")
    
    return {
        'models': models,
        'ranking': sorted_models,
        'summary': {
            'best_model': sorted_models[0][1]['name'] if sorted_models else None,
            'best_score': sorted_models[0][1]['final_avg'] if sorted_models else None,
            'total_models': len(models)
        }
    }

