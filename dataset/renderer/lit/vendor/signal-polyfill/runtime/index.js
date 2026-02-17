class State {
  constructor(value) {
    this._value = value;
  }

  get() {
    return this._value;
  }

  set(value) {
    this._value = value;
  }
}

class Computed {
  constructor(compute) {
    this._compute = typeof compute === "function" ? compute : () => undefined;
  }

  get() {
    return this._compute();
  }
}

class Watcher {
  constructor(onSignalChanged) {
    this._onSignalChanged =
      typeof onSignalChanged === "function" ? onSignalChanged : () => {};
  }

  watch(_signal) {
    return;
  }

  unwatch(_signal) {
    return;
  }
}

const subtle = {
  Watcher,
  untrack(callback) {
    return typeof callback === "function" ? callback() : callback;
  },
};

const Signal = {
  State,
  Computed,
  subtle,
};

export { Signal, State, Computed, subtle };
