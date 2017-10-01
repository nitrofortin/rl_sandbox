import numpy as np
discrete_action = True

class DeterministicLinearPolicy(object):

    def __init__(self, theta, dim_ob, n_actions):
        """
        dim_ob: dimension of observations
        n_actions: number of actions
        theta: flat vector of parameters
        """

        self.W = theta[:,:n_actions]

        self.b = theta[:,n_actions : None].reshape(dim_ob, n_actions)

    def act(self, ob):
        ob = np.array([ob,ob]).reshape(ob.shape[0],ob.shape[0]*2)
        y = ob.dot(self.W) + self.b

        if discrete_action:
            a = y.argmax(axis=1)
            return np.array([[1-i,i] for i in a]).T
        else:
            partition = np.exp(y)/np.array([np.exp(y).sum(axis=1),np.exp(y).sum(axis=1)]).T
            return partition.T

class Graph(object):
    def __init__(self, adj_matrix):
        self.adj_matrix = adj_matrix
        
    def step(self, partition):
        deuxM = self.adj_matrix.sum()
        k = np.array([self.adj_matrix.sum(axis=0)]).astype(np.float32)
        modularite = self.adj_matrix - k.T.dot(k)/deuxM
        return np.trace(partition.dot(modularite).dot(partition.T))


def do_episode(policy, env, num_steps, render=False):
    total_rew = 0
    ob = env.adj_matrix
    for t in range(num_steps):
        partition = policy.act(ob)
        reward = env.step(partition)
        total_rew += reward
    return total_rew/num_steps

def noisy_evaluation(theta):
    policy = make_policy(theta)
    rew = do_episode(policy, env, num_steps)
    return rew

def make_policy(theta):
    return DeterministicDiscreteActionLinearPolicy(theta, dim_ob, n_actions)


if __name__ == "__main__":
    adj_matrix = np.array([[0,1,1,0,0,0],
                           [1,0,1,0,0,0],
                           [1,1,0,1,0,0],
                           [0,0,1,0,1,1],
                           [0,0,0,1,0,1],
                           [0,0,0,1,1,0]])

    env = Graph(adj_matrix)

    n_clusters = 2
    dim_ob = adj_matrix.shape[0]
    n_actions = n_clusters

    num_steps = 500 # maximum length of episode
    n_iter = 15 # number of iterations of cross-entropy method
    batch_size = 100 # number of samples per batch
    elite_frac = 0.05

    dim_theta = dim_ob * n_actions

    theta_mean = np.zeros((dim_theta, n_actions + 1))
    theta_std = np.ones((dim_theta, n_actions + 1))

    for iteration in xrange(n_iter):
        thetas = [np.random.normal(theta_mean,theta_std,(dim_theta, n_actions + 1)) for i in range(batch_size)]
        rewards = [noisy_evaluation(theta) for theta in thetas]
        n_elite = int(batch_size * elite_frac)
        elite_inds = np.argsort(rewards)[batch_size - n_elite:batch_size]
        elite_thetas = [thetas[i] for i in elite_inds]
        theta_mean = np.array(elite_thetas).mean(axis=0)
        theta_std = np.array(elite_thetas).std(axis=0)
        MEAN.append(theta_mean)
        STD.append(theta_std)
        print "iteration %i. mean f: %8.3g. max f: %8.3g"%(iteration, np.mean(rewards), np.max(rewards))
        do_episode(make_policy(theta_mean), env, num_steps, render=True)




