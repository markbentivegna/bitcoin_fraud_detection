import torch
from torch import nn
import collections
import tqdm
import numpy as np
import time
from models.GraphGANGenerator import GraphGANGenerator
from models.GraphGANDiscriminator import GraphGANDiscriminator

class GraphGAN(nn.Module):
    def __init__(self, node_count, graph, initial_node_embedding_generator, initial_node_embedding_discriminator):
        super().__init__()
        self.node_count = node_count
        self.root_nodes = [i for i in range(node_count)]
        self.graph = self.generate_node_mappings(graph)

        self.generator = GraphGANGenerator(node_count, initial_node_embedding_generator)
        self.discriminator = GraphGANDiscriminator(node_count, initial_node_embedding_discriminator)
        self.trees = self.construct_trees(self.root_nodes)

    def generate_node_mappings(self, graph):
        node_mappings = {}
        nodes = graph.x[:,0].to(torch.int32)
        for node in nodes:
            edges = graph.edge_index[graph.edge_index[:,0] == node]
            children = edges[:,1]
            node_mappings[int(node)] = list(children.numpy().astype(int))
        return node_mappings


    def construct_trees(self, nodes):
        trees = {}
        for root in tqdm.tqdm(nodes):
            trees[root] = {}
            
            trees[root][root] = [root]

            visited_nodes = set()
            queue = collections.deque([root])
            while len(queue) > 0:
                current_node = queue.popleft()
                visited_nodes.add(current_node)
                for child_node in self.graph[int(current_node)]:
                    if child_node not in visited_nodes:
                        trees[root][current_node].append(child_node)
                        trees[root][child_node] = [current_node]
                        queue.append(child_node)
                        visited_nodes.add(child_node)
        return trees
    
    def prepare_data_for_discriminator(self):
        center_nodes, neighbor_nodes, labels = [], [], []

        for i in self.root_nodes:
            if np.random.rand() < 1:
                positive_sample = self.graph[int(i)]
                iter_start = time.time()
                negative_sample, _ = self.sample(i, self.trees[i], len(positive_sample), for_discriminator=True)
                iter_end = time.time()
                print(f"Iteration {i} took {iter_end - iter_start} seconds to complete")
                if len(positive_sample) != 0 and negative_sample is not None:
                    center_nodes.extend([i] * len(positive_sample))
                    neighbor_nodes.extend(positive_sample)
                    labels.extend([1] * len(positive_sample))

                    center_nodes.extend([i] * len(positive_sample))
                    neighbor_nodes.extend(negative_sample)
                    labels.extend([0]*len(negative_sample))
        return center_nodes, neighbor_nodes, labels

    def prepare_data_for_generator(self):
        paths = []
        for i in self.root_nodes:
            if np.random.rand() < 1:
                sample, paths_from_index = self.sample(i, self.trees[i], 20, for_d=False)
                if paths_from_index is not None:
                    paths.extend(paths_from_index)
        node_pairs = list(map(self.get_node_pairs_from_path, paths))
        node_1, node_2 = [], []
        for i in range(len(node_pairs)):
            for pair in node_pairs[i]:
                node_1.append(pair[0])
                node_2.append(pair[1])
        reward = self.sess.run(self.discriminator.reward,
                               feed_dict={self.discriminator.node_id: np.array(node_1),
                                          self.discriminator.node_neighbor_id: np.array(node_2)})
        return node_1, node_2, reward

    def sample(self, root, tree, sample_count, for_discriminator):
        graph_score = self.generator.get_all_score()
        samples, paths = [], []
        n = 0

        while len(samples) < sample_count:
            current_node = root
            previous_node = -1
            paths.append([])
            is_root = True
            paths[n].append(current_node)
            while True:
                node_neighbor = tree[current_node][1:] if is_root else tree[current_node]
                is_root = False
                if len(node_neighbor) == 0:
                    return None, None
                if for_discriminator:
                    if node_neighbor == [root]:
                        return None, None
                    if root in node_neighbor:
                        node_neighbor.remove(root)
                relevance_score = graph_score[current_node, node_neighbor]
                relevance_score = self._softmax(relevance_score.detach().numpy())
                next_node = np.random.choice(node_neighbor, size=1, p=relevance_score)[0]
                paths[n].append(next_node)
                if next_node == previous_node:
                    samples.append(current_node)
                    break
                previous_node = current_node
                current_node = next_node
            n = n + 1
        return samples, paths

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def train(self, device):        
        all_score = torch.matmul(self.generator.embedding_matrix, self.generator.embedding_matrix.t()) + self.generator.bias_vector
        criterion1 = nn.BCEWithLogitsLoss()

        discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=1e-3)
        generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-3)

        for epoch in range(20):
            print("epoch %d" % epoch)

            if epoch > 0 and epoch % 10 == 0:
                torch.save(self.generator.state_dict(), 'saved_models/generator_checkpoint.pt')
                torch.save(self.discriminator.state_dict(), 'saved_models/discriminator_checkpoint.pt')

            # D-steps
            center_nodes = []
            neighbor_nodes = []
            labels = []
            for discriminator_epoch in range(30):
                if discriminator_epoch % 30 == 0:
                    center_nodes, neighbor_nodes, labels = self.prepare_data_for_discriminator()

                train_size = len(center_nodes)
                start_list = list(range(0, train_size, 64))
                np.random.shuffle(start_list)
                for start in start_list:
                    end = start + 64

                    discriminator_optimizer.zero_grad()
                    score, node_embedding, node_neighbor_embedding, bias = self.discriminator(np.array(center_nodes[start:end]),
                    np.array(neighbor_nodes[start:end]))
                    label = np.array(labels[start:end])
                    label = torch.from_numpy(label).type(torch.float64)
                    label = label.to(device)

                    discriminator_loss = torch.sum(criterion1(score, label)) + 1e-5 * (torch.sum(node_embedding**2)/2+
                    torch.sum(node_neighbor_embedding**2)/2+torch.sum(bias**2)/2)

                    discriminator_loss.backward()
                    discriminator_optimizer.step()

                print(f"[Total Epoch {epoch}/{20}] [D Epoch {discriminator_epoch}/{30}] [D loss: {discriminator_loss.item():.4f}]")
                
                if discriminator_epoch == 30 - 1:
                    print(f"Discrimination finished(Epoch {epoch}).")

            node_1 = []
            node_2 = []
            reward = []
            for g_epoch in range(30):
                all_score = self.generator.get_all_score()
                if g_epoch % 30 == 0:
                    node_1, node_2, reward = self.prepare_data_for_generator(self.discriminator, self.root_nodes, self.trees, all_score)
                    reward = reward.detach()

                # training
                train_size = len(node_1)
                start_list = list(range(0, train_size, 64))
                np.random.shuffle(start_list)
                for start in start_list:
                    end = start + 64

                    generator_optimizer.zero_grad()
                    node_embedding, node_neighbor_embedding, prob = self.generator(np.array(node_1[start:end]), np.array(node_2[start:end]))
                    reward_p = reward[start:end]

                    generator_loss = -torch.mean(torch.log(prob)*reward_p) + 1e-5 * (torch.sum(node_embedding**2)/2+
                    torch.sum(node_neighbor_embedding**2)/2)

                    generator_loss.backward()
                    generator_optimizer.step()

                print(f"[Total Epoch {epoch}/{20}] [G Epoch {g_epoch}/{30}] [G loss: {generator_loss.item():.4f}]")


                if g_epoch == 30 - 1:
                    print(f"Generation finished (Epoch {epoch}).")


            # self.write_embeddings_to_file(self.generator, self.discriminator, self.node_count)
            # self.evaluation(self.node_count)
        
        print('Training completes')

    def forward(self):
        pass