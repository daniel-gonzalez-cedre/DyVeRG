from typing import Tuple, List, Set, Union, Any
from collections import deque
from sys import setrecursionlimit
setrecursionlimit(1000)


class TreeNode:
    """
    Node class for trees
    """
    __slots__ = 'key', 'level', 'children', 'leaves', 'parent', 'kids', 'is_leaf'

    def __init__(self, key: str, is_leaf: bool = False) -> None:
        self.key = key   # key of the node, each node has an unique key
        self.level = 0  # level of the node

        self.children: Set[Union[int, str]] = set()  # set of node labels of nodes in the subtree rooted at the node
        self.leaves: Set[int] = set()  # set of children that are leaf nodes

        self.parent: Union[TreeNode, None] = None  # pointer to parent
        self.kids: List[TreeNode] = []  # pointers to the children

        self.is_leaf: bool = is_leaf  # True if it's a child, False otherwise

    def __eq__(self, other) -> bool:
        if other is None:
            return False
        elif isinstance(other, str) or isinstance(other, int):
            return self.key == other
        elif isinstance(other, TreeNode):
            if self.key != other.key:
                return False

            if len(self.kids) != len(other.kids) or self.children != other.children:
                return False

            if {kid.key for kid in self.kids} != {kid.key for kid in other.kids}:
                return False

            for subtree, othertree in {(selfkid, otherkid) for selfkid in self.kids for otherkid in other.kids
                                       if selfkid.key == otherkid.key}:
                if subtree != othertree:
                    return False

            return True
        else:
            raise TypeError(f'TeeNode equality is not supported for objects of type {type(other)} such as {other}.')

    def __str__(self) -> str:
        if self.parent is None:
            parent = None
        else:
            parent = self.parent.key

        return f'{self.key} ({len(self.leaves)}) p: {parent}'

    def __repr__(self) -> str:
        return f'{self.key} ({len(self.leaves)})'

    # deep copy
    def __copy__(self):
        node = TreeNode(key=self.key)
        node.children = self.children.copy()
        node.leaves = self.leaves.copy()
        node.is_leaf = self.is_leaf
        node.level = self.level

        for kid in self.kids:
            node.kids.append(kid.copy())

        for kid in node.kids:
            kid.parent = node

        return node
    # def __copy__(self):
    #     node_copy = TreeNode(key=self.key)
    #     node_copy.parent = self.parent
    #     node_copy.kids = self.kids
    #     node_copy.leaves = self.leaves
    #     node_copy.children = self.children
    #     node_copy.level = self.level
    #     node_copy.is_leaf = self.is_leaf
    #     return node_copy

    def __hash__(self):
        return hash(self.key)

    def copy(self):
        return self.__copy__()

    def make_leaf(self, new_key) -> None:
        """
        converts the internal tree node into a leaf
        :param new_key: new key of the node
        :return:
        """
        self.leaves = {self.key}  # update the leaves
        self.children = set()
        self.kids = []
        self.is_leaf = True
        self.key = new_key

    def get_num_leaves(self) -> int:
        return len(self.leaves)


def create_tree(lst: List[Any]) -> TreeNode:
    """
    Creates a Tree from the list of lists
    :param lst: nested list of lists
    :return: root of the tree
    """
    key = '0'

    def create(lst):
        nonlocal key

        if len(lst) == 1 and isinstance(lst[0], int):  # detect leaf
            return TreeNode(key=lst[0], is_leaf=True)
        node = TreeNode(key=key)
        # key = chr(ord(key) + 1)
        key = str(int(key) + 1)

        for item in lst:
            node.kids.append(create(item))

        return node

    root = create(lst)

    def update_info(node) -> Tuple[Set[Union[int, str]], int]:
        """
        updates the parent pointers, payloads, and the number of leaf nodes
        :param node:
        :return:
        """
        if node.is_leaf:
            node.make_leaf(new_key=node.key)  # the key doesn't change
        else:
            for kid in node.kids:
                kid.parent = node
                kid.level = node.level + 1
                children, leaves = update_info(kid)
                node.children.add(kid.key)
                node.children.update(children)
                node.leaves.update(leaves)

        return node.children, node.leaves

    def relabel_tnodes(tnode):
        q = deque()
        q.append(tnode)
        key = '0'
        while len(q) != 0:
            tnode = q.popleft()
            if tnode.is_leaf:
                continue
            tnode.key = key
            # key = chr(ord(key) + 1)
            key = str(int(key) + 1)
            for kid in tnode.kids:
                q.append(kid)

    relabel_tnodes(root)
    update_info(node=root)

    return root
