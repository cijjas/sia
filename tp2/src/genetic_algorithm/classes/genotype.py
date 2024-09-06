class Genotype:

    def __init__ (self, strength, dexterity, intelligence, vigor, constitution, height):
        self.strength = strength
        self.dexterity = dexterity
        self.intelligence = intelligence
        self.vigor = vigor
        self.constitution = constitution
        self.height = height
        self.attributes = [strength, dexterity, intelligence, vigor, constitution, height]


    def __eq__(self, other):
        return self.attributes == other.attributes

    def from_array(cls, array):
        return cls(*array)

    def __getitem__(self, index):
        return self.attributes[index]

    def __setitem__(self, index, value):
            self.attributes[index] = value
            self.strength, self.dexterity, self.intelligence, self.vigor, self.constitution, self.height = self.attributes

    def get_strength(self):
        return self.strength
    def get_dexterity(self):
        return self.dexterity
    def get_intelligence(self):
        return self.intelligence
    def get_vigor(self):
        return self.vigor
    def get_constitution(self):
        return self.constitution
    def get_height(self):
        return self.height

    def set_strength(self, strength):
        self.strength = strength
    def set_dexterity(self, dexterity):
        self.dexterity = dexterity
    def set_intelligence(self, intelligence):
        self.intelligence = intelligence
    def set_vigor(self, vigor):
        self.vigor = vigor
    def set_constitution(self, constitution):
        self.constitution = constitution
    def set_height(self, height):
        self.height = height

    def as_array(self):
        return [self.strength, self.dexterity, self.intelligence, self.vigor, self.constitution, self.height]

    def as_dict(self):
        return {
            "strength": self.strength,
            "dexterity": self.dexterity,
            "intelligence": self.intelligence,
            "vigor": self.vigor,
            "constitution": self.constitution,
            "height": self.height
        }

    def get_total_points(self):
        return sum(self.attributes[:-1])

    def __len__(self):
        return len(self.attributes)

    def __str__(self):
        return f"Genotype: {self.strength}, {self.dexterity}, {self.intelligence}, {self.vigor}, {self.constitution}, {self.height}"