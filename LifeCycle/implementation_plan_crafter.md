# Plan d'Implémentation : Spécialisation des Neurones INN (Crafter)

## Objectif
Restructurer `INN_Brain` pour avoir une topologie spécialisée plutôt qu'une population homogène "tout-vers-tout".

## 1. Nouvelle Topologie
Au lieu de 6 neurones identiques recevant tous un mix d'inputs, nous allons définir 3 types :
*   **Sensor Neurons (N_SENSORS=2)** :
    *   Neuron 0 : Reçoit UNIQUEMENT l'embedding `Environment` (Vue large).
    *   Neuron 1 : Reçoit UNIQUEMENT l'embedding `Inventory` (Vue détaillée).
*   **Reflection Neurons (N_REFLECT=3)** :
    *   Ne reçoivent **aucun input externe direct**.
    *   Leur but est de processer l'histoire et de "réfléchir" en attendant le prochain pas.
    *   Ils lisent l'état des Sensor Neurons via Attention.
*   **Action Neuron (N_ACTION=1)** :
    *   C'est le "Chef d'Orchestre".
    *   Il lit tout le monde (Sensor + Reflection).
    *   C'est **le seul** à être connecté aux têtes `Actor` (Logits) et `Critic` (Value).

## 2. Modifications Techniques (`crafter_inn_benchmark.py`)

### A. Classe `INN_Brain`
*   Supprimer `gate_logits` (plus besoin de mixer dynamiquement, le routage est structurel).
*   Modifier la boucle de `forward` pour traiter les groupes séparément :
    1.  **Sensors Update** : Update `h_env` avec input `env_feat` et `h_inv` avec `inv_feat`.
    2.  **Global Attention** : Tout le monde peut lire tout le monde (ou *causal* : Action lit tout, Reflection lit Sensors ? **Décision :** Laissons l'Attention "Tous-vers-Tous" mais l'injection d'input est localisée. Les neurones de réflexion "verront" les sensors via l'attention).
    3.  **Action Readout** : Au lieu de `torch.mean(all_states)`, on prend `state[action_neuron_idx]`.

### B. Classe `INN_Neuron`
*   Ajouter un flag ou argument `input_dim` dans `forward`. Si `None`, l'input externe est 0.

### C. Hyperparamètres
*   `N_NEURONS` = 6 (reste pareil : 1 Env + 1 Inv + 3 Refl + 1 Action).

## 3. Avantages Attendus
1.  **Interpretability** : On pourra voir si le neurone "Inventaire" s'active quand on ramasse du bois.
2.  **Stabilité** : Le neurone d'action ne sera pas "pollué" directement par le bruit visuel à chaque frame, il doit passer par le filtrage de l'attention.
3.  **Correspondance Utilisateur** : Colle exactement à la demande.

## 4. Code Changes Details
```python
# Pseudo-code structure change
class INN_Brain(nn.Module):
    # ...
    def forward(self, obs, state, ...):
        # 1. Encode
        inv, env = self.dual_encoder(x)
        
        # 2. Iterate Neurons
        new_states = []
        for i in range(self.n_neurons):
            ext_input = 0
            if i == 0: ext_input = env  # Sensor Env
            elif i == 1: ext_input = inv # Sensor Inv
            # Else (Reflection/Action): ext_input = 0
            
            # Attention Mechanism (Global Workspace)
            # ...
            # RNN Update
            # ...
            
        # 3. Readout
        # ONLY Action Neuron (Last one)
        action_state = new_states[-1] 
        logits = self.actor(action_state)
```
